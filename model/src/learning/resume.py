"""Validate and score saved adapters before restarting a training process."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .config import MAX_STEPS_PER_ROUND
from .scoring import VALIDATION_LABEL, evaluate_split
from .scoring.discovery import (
    adapter_directory,
    checkpoint_directory,
    checkpoint_run_dir,
    checkpoint_steps,
)
from .training.checkpoints import Selection, checkpoint_step


def validate_resume_checkpoint(path: str) -> str:
    """Require an indexed adapter checkpoint and compatible round metadata."""
    root = Path(__file__).resolve().parents[2]
    requested = Path(path)
    resolved = (
        root / requested if not requested.is_absolute() else requested
    ).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise SystemExit(
            "[!] Resume checkpoint must stay inside the model workspace."
        ) from None
    discovered = adapter_directory(os.fspath(resolved))
    adapter_dir = Path(discovered) if discovered is not None else resolved
    checkpoint_dir = Path(checkpoint_directory(os.fspath(adapter_dir)))
    run_dir = Path(checkpoint_run_dir(os.fspath(adapter_dir)))
    if not (run_dir / "run_metadata.json").is_file():
        raise SystemExit(f"[!] Resume run {run_dir} has no valid run metadata.")
    try:
        metadata = json.loads(
            Path(run_dir, "run_metadata.json").read_text(encoding="utf-8")
        )
        stride = int(metadata["max_steps_per_round"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise SystemExit(
            f"[!] Resume run {run_dir} has no valid run metadata."
        ) from None
    if stride != MAX_STEPS_PER_ROUND:
        raise SystemExit(
            f"[!] Resume run stride {stride} differs from configured {MAX_STEPS_PER_ROUND}."
        )
    indexed_path = os.path.normcase(os.path.realpath(adapter_dir))
    indexed = any(
        os.path.normcase(os.path.realpath(saved)) == indexed_path
        for _step, saved in checkpoint_steps(os.fspath(run_dir))
    )
    local_step = checkpoint_step(str(checkpoint_dir))
    if (
        local_step is None
        or local_step < 1
        or local_step > stride
        or not checkpoint_dir.is_dir()
        or not adapter_dir.is_dir()
        or not indexed
        or not (adapter_dir / "adapter_config.json").is_file()
        or not any(
            (adapter_dir / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        )
    ):
        raise SystemExit(
            f"[!] Resume checkpoint {resolved} is incomplete or unindexed."
        )
    return os.fspath(adapter_dir)


def remaining_training_seconds(run_dir: str, max_hours: float) -> float:
    """Preserve the existing run's cap, using metadata creation time as a fallback."""
    path = Path(run_dir, "run_metadata.json")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        started = float(metadata.get("started_at", path.stat().st_mtime))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        raise SystemExit(
            "[!] Resume run has no valid original training timestamp."
        ) from None
    return max(0.0, started + max_hours * 3600 - time.time())


def next_round_number(run_dir: str) -> int:
    """Return the highest round directory, including empty crash leftovers."""
    pattern = re.compile(r"^round-(\d+)$")
    try:
        numbers = (
            int(match.group(1))
            for name in os.listdir(run_dir)
            if (match := pattern.fullmatch(name))
            and os.path.isdir(os.path.join(run_dir, name))
        )
        return max(numbers, default=0)
    except OSError as exc:
        raise SystemExit(f"[!] Cannot inspect resume run {run_dir!r}: {exc}") from None


def score_resume_checkpoint(
    dependencies: dict[str, Any],
    model,
    tokenizer,
    validation_rows: list[dict[str, Any]],
    checkpoint_path: str,
    limit: int,
    deadline: float,
) -> Selection:
    """Rescore the requested adapter on validation before restart training."""
    adapter_path = adapter_directory(checkpoint_path) or checkpoint_path
    run_dir = checkpoint_run_dir(adapter_path)
    normalized = os.path.normcase(os.path.realpath(adapter_path))
    step = next(
        (
            logical
            for logical, saved in checkpoint_steps(run_dir)
            if os.path.normcase(os.path.realpath(saved)) == normalized
        ),
        None,
    )
    if step is None:
        raise SystemExit(
            f"[!] Resume checkpoint {checkpoint_path!r} is not indexed by its run."
        )
    checkpoint_path = adapter_path
    name = f"resume-score-{step}"
    original = getattr(model, "active_adapter", "default")
    if callable(original):
        original = original()
    if isinstance(original, list):
        original = original[0]
    loaded = False
    try:
        model.load_adapter(checkpoint_path, adapter_name=name, is_trainable=False)
        loaded = True
        model.set_adapter(name)
        result = evaluate_split(
            model,
            tokenizer,
            validation_rows,
            limit,
            split=VALIDATION_LABEL,
            deadline=deadline,
        )
        result.pop("predictions", None)
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(
            f"[!] Resume adapter could not be validation-scored ({type(exc).__name__}: {exc}); "
            "no test scoring or deployment."
        ) from None
    finally:
        if loaded:
            model.set_adapter(original)
            model.delete_adapter(name)
        dependencies["torch"].cuda.empty_cache()
    return Selection(step, None, False, result, checkpoint_path)
