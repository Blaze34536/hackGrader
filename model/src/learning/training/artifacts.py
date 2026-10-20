"""Manage run checkpoints, artifact stamps, and conservative runtime estimates."""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import statistics
import time
from typing import Any

from ..config import (
    ACCUM,
    ASSUMED_SECONDS_PER_STEP,
    BASE_MODEL,
    BUDGET_SAFETY_FACTOR,
    EPOCHS,
    HOURS,
    MAX_LEN,
    MAX_STEPS_PER_ROUND,
    MAX_TRAINING_HOURS,
    MICRO_BATCH,
)
from ..deployment.provenance import weight_time
from .trainer_config import configure_evaluation, eval_configuration_fields

__all__ = ["configure_evaluation", "eval_configuration_fields"]


class RunDeadlineExceeded(RuntimeError):
    """Raised when a bounded training or scoring phase reaches its deadline."""


def require_time_remaining(deadline: float | None, phase: str) -> None:
    """Stop a phase before it begins or continues beyond the absolute deadline."""
    if deadline is not None and time.monotonic() >= deadline:
        raise RunDeadlineExceeded(f"the {phase} deadline was reached")


def initialize_run_directory(path: str, *, forked_from: str | None = None) -> None:
    """Create a fresh run folder and record its stride and optional source checkpoint."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    os.mkdir(path)
    metadata = {
        "max_steps_per_round": MAX_STEPS_PER_ROUND,
        "started_at": time.time(),
        "base_model": BASE_MODEL,
        "max_len": MAX_LEN,
    }
    if forked_from is not None:
        run_parent = os.path.dirname(os.path.abspath(path)) or "."
        metadata["forked_from"] = os.path.relpath(forked_from, run_parent)
    with open(os.path.join(path, "run_metadata.json"), "x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def verify_merge(path: str, adapter_path: str) -> list[str]:
    """Ensure merged weights exist and were written or created after the adapter."""
    shards = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not shards:
        raise SystemExit(
            f"[!] {path} has no merged safetensors weights; deployment stopped."
        )
    adapter_time = weight_time(adapter_path)
    stale = [shard for shard in shards if weight_time(shard) < adapter_time]
    if stale:
        raise SystemExit(
            f"[!] {len(stale)} merged shards predate the adapter; deployment stopped."
        )
    return shards


def write_stamp(
    path: str, adapter_path: str, training_rows: int, base_model: str, max_len: int
) -> dict[str, Any]:
    """Record the selected adapter hash and training settings beside merged weights."""
    digest = hashlib.sha256()
    with open(adapter_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    stamp = {
        "adapter_sha256": digest.hexdigest(),
        "adapter_path": adapter_path,
        "base_model": base_model,
        "training_rows": training_rows,
        "max_len": max_len,
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(path, "merge_stamp.json"), "w", encoding="utf-8") as handle:
        json.dump(stamp, handle, indent=2)
    return stamp


def measured_seconds_per_step() -> tuple[float, str] | None:
    """Estimate recent step time from round-local checkpoint timestamps."""
    grouped: dict[str, list[tuple[int, float]]] = {}
    pattern = os.path.join("outputs", "*", "**", "checkpoint-*", "trainer_state.json")
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, encoding="utf-8") as handle:
                steps = int(json.load(handle).get("global_step") or 0)
            modified = os.path.getmtime(path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if steps > 0:
            round_dir = os.path.dirname(os.path.dirname(path))
            grouped.setdefault(round_dir, []).append((steps, modified))
    newest = sorted(
        grouped.items(),
        key=lambda item: max(timestamp for _step, timestamp in item[1]),
        reverse=True,
    )
    for round_dir, states in newest:
        states.sort()
        rates = [
            (later[1] - earlier[1]) / (later[0] - earlier[0])
            for earlier, later in itertools.pairwise(states)
            if later[0] > earlier[0] and later[1] > earlier[1]
        ]
        if rates:
            return statistics.median(rates), os.path.basename(round_dir)
    return None


def report_training_rate(trainer) -> None:
    """Report elapsed time per completed optimizer step."""
    history = getattr(trainer.state, "log_history", None) or []
    runtime = next(
        (
            entry["train_runtime"]
            for entry in reversed(history)
            if "train_runtime" in entry
        ),
        None,
    )
    steps = int(getattr(trainer.state, "global_step", 0) or 0)
    if runtime and steps:
        print(
            f"[i] trained {steps} step(s) in {runtime / 3600:.2f}h, {runtime / steps:.1f}s per step"
        )


def report_stale_artifacts() -> None:
    """Report old checkpoints without removing prior run artifacts."""
    leftovers = sorted(glob.glob("outputs/checkpoint-*"))
    if leftovers:
        print(f"[i] {len(leftovers)} older checkpoint director(ies) remain untouched.")


def estimate_run_hours(
    train_count: int,
    seconds_per_step: float,
    *,
    batch: int,
    accumulation: int,
    epochs: int,
) -> float:
    """Estimate one configured training pass from rows and optimizer throughput."""
    if batch <= 0 or accumulation <= 0 or epochs <= 0 or seconds_per_step <= 0:
        raise ValueError("batch, accumulation, epochs, and step time must be positive")
    steps = -(-train_count // (batch * accumulation)) * epochs
    return steps * seconds_per_step / 3600


def check_training_budget(train_count: int) -> None:
    """Report one training-round estimate against the hard continuation cap."""
    if int(os.environ.get("HACKGRADER_MAX_STEPS", "0")) > 0:
        raise SystemExit(
            "[!] HACKGRADER_MAX_STEPS is a smoke-test setting, not a release run."
        )
    if HOURS <= 0 or HOURS > MAX_TRAINING_HOURS:
        raise SystemExit(
            f"[!] HACKGRADER_HOURS must be positive and no more than "
            f"{MAX_TRAINING_HOURS:g}; training was not started."
        )
    if BUDGET_SAFETY_FACTOR < 1:
        raise SystemExit(
            "[!] HACKGRADER_BUDGET_SAFETY_FACTOR must be at least 1; no training started."
        )
    measured = measured_seconds_per_step()
    seconds, origin = measured or (ASSUMED_SECONDS_PER_STEP, "assumed for this dataset")
    estimate = estimate_run_hours(
        train_count,
        seconds * BUDGET_SAFETY_FACTOR,
        batch=MICRO_BATCH,
        accumulation=ACCUM,
        epochs=EPOCHS,
    )
    print(
        f"[i] estimated full run: {estimate:.1f}h from {origin} at {seconds:.1f}s/step "
        f"plus a {BUDGET_SAFETY_FACTOR:.2f}x margin; overall cap "
        f"{HOURS:.1f}h (hard maximum {MAX_TRAINING_HOURS:.1f}h)."
    )
    if estimate > HOURS:
        print("[!] One training round is estimated to exceed the cap.")
