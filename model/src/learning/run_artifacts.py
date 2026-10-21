"""Read what one run directory holds without loading a model.

A run's next step depends on facts that live on disk: whether it claimed its one-time
independent test, which checkpoints it saved, whether an adapter was written after that
claim, and whether a merge stamp matches the released model. This module only reads
those facts.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

ATTEMPT_MARKER = ".independent-test-attempted"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
STAMP_NAME = "merge_stamp.json"
RELEASED_MODEL = os.path.join("data", "deployed_model.json")
DIAGNOSTICS_PREFIX = "validation-checkpoint-"
RUN_ID_LENGTH = 15


@dataclass(frozen=True)
class RunState:
    """What one run directory holds now, read without loading a model."""

    run_id: str
    run_dir: str
    started_at: float | None
    claimed_at: float | None
    adapter_dir: str
    adapter_file: str | None
    adapter_written_at: float | None
    merge_dir: str
    merged: bool
    deployed: bool
    checkpoints: tuple[tuple[int, str], ...]
    selected: tuple[int, str] | None
    inputs: tuple[str, ...]


def outputs_root() -> str:
    """Return the configured directory that holds run subdirectories."""
    configured = os.environ.get("HACKGRADER_OUTPUT_DIR")
    return os.path.dirname(configured) if configured else "outputs"


def run_ids(root: str | None = None) -> list[str]:
    """List run directories oldest first, ignoring anything without run metadata."""
    root = root or outputs_root()
    try:
        names = os.listdir(root)
    except OSError:
        return []
    return sorted(
        name
        for name in names
        if _is_run_id(name)
        and os.path.isfile(os.path.join(root, name, "run_metadata.json"))
    )


def continuable_run(root: str | None = None) -> str | None:
    """Return the newest run that holds a checkpoint or a spent independent test.

    A run that stopped before its first checkpoint cannot be continued, so continuing
    the newest directory would report nothing to do while an older run still needs a
    resume or a rebuilt release.
    """
    root = root or outputs_root()
    found = run_ids(root)
    for run_id in reversed(found):
        run_dir = os.path.join(root, run_id)
        if os.path.isdir(os.path.join(run_dir, ATTEMPT_MARKER)):
            return run_id
        metadata = _read_json(os.path.join(run_dir, "run_metadata.json"))
        if _checkpoint_entries(run_dir, metadata):
            return run_id
    return found[-1] if found else None


def newest_run(root: str | None = None) -> str | None:
    """Return the most recent run that wrote its metadata."""
    found = run_ids(root)
    return found[-1] if found else None


def newest_claimed_run(root: str | None = None) -> str | None:
    """Return the most recent run that already claimed its independent test."""
    root = root or outputs_root()
    for run_id in reversed(run_ids(root)):
        if os.path.isdir(os.path.join(root, run_id, ATTEMPT_MARKER)):
            return run_id
    return None


def inspect(
    run_id: str,
    *,
    root: str | None = None,
    adapter_dir: str | None = None,
    merge_dir: str | None = None,
    released: str = RELEASED_MODEL,
) -> RunState:
    """Read the facts a continuation decision depends on."""
    run_dir = os.path.join(root or outputs_root(), run_id)
    adapter_dir = adapter_dir or os.path.join("outputs", f"lora_adapter-{run_id}")
    merge_dir = merge_dir or os.path.join("models", f"qwen3_ai_detector_merged-{run_id}")
    metadata = _read_json(os.path.join(run_dir, "run_metadata.json"))
    marker = os.path.join(run_dir, ATTEMPT_MARKER)
    stamped = _read_json(os.path.join(merge_dir, STAMP_NAME)).get("adapter_sha256")
    saved = _read_json(released).get("adapter_sha256")
    adapter_file = os.path.join(adapter_dir, ADAPTER_WEIGHTS)
    if not os.path.isfile(adapter_file):
        adapter_file = None
    checkpoints = _checkpoint_entries(run_dir, metadata)
    return RunState(
        run_id=run_id,
        run_dir=run_dir,
        started_at=_number(metadata.get("started_at")),
        claimed_at=os.path.getmtime(marker) if os.path.isdir(marker) else None,
        adapter_dir=adapter_dir,
        adapter_file=adapter_file,
        adapter_written_at=(
            os.path.getmtime(adapter_file) if adapter_file is not None else None
        ),
        merge_dir=merge_dir,
        merged=bool(stamped),
        deployed=bool(stamped and stamped == saved),
        checkpoints=checkpoints,
        selected=_selected_checkpoint(run_dir, checkpoints),
        inputs=recorded_inputs(metadata),
    )


def recorded_inputs(metadata: dict[str, object]) -> tuple[str, ...]:
    """Return the data files a run trained on, as recorded when it started.

    A resume must keep training on the same rows the run started with, so the run
    records the dataset it was given. Runs that predate that record have none here and
    fall back to the configured inputs.
    """
    recorded = metadata.get("inputs")
    if not isinstance(recorded, list) or not all(
        isinstance(path, str) for path in recorded
    ):
        return ()
    return tuple(path for path in recorded if path != "Modelfile")


def _checkpoint_entries(
    run_dir: str, metadata: dict[str, object]
) -> tuple[tuple[int, str], ...]:
    """List saved adapters by logical step, oldest first."""
    stride = _number(metadata.get("max_steps_per_round"))
    found: dict[int, str] = {}
    try:
        names = sorted(os.listdir(run_dir))
    except OSError:
        return ()
    for name in names:
        step = _numbered(name, "checkpoint-")
        if step is not None:
            adapter = _adapter_in(os.path.join(run_dir, name))
            if adapter is not None:
                found[step] = adapter
    if stride is None or stride < 1:
        return tuple(sorted(found.items()))
    stride = int(stride)
    for name in names:
        round_number = _numbered(name, "round-")
        round_dir = os.path.join(run_dir, name)
        if round_number is None or not os.path.isdir(round_dir):
            continue
        for child in sorted(os.listdir(round_dir)):
            step = _numbered(child, "checkpoint-")
            if step is None:
                continue
            adapter = _adapter_in(os.path.join(round_dir, child))
            if adapter is not None:
                found[(round_number - 1) * stride + step] = adapter
    return tuple(sorted(found.items()))


def _selected_checkpoint(
    run_dir: str, checkpoints: tuple[tuple[int, str], ...]
) -> tuple[int, str] | None:
    """Recover the checkpoint this run's own validation sweep selected."""
    saved = dict(checkpoints)
    best: tuple[tuple[float, ...], int] | None = None
    try:
        names = sorted(os.listdir(run_dir))
    except OSError:
        return None
    for name in names:
        if not (name.startswith(DIAGNOSTICS_PREFIX) and name.endswith(".jsonl")):
            continue
        recorded = _first_json(os.path.join(run_dir, name)).get("metadata")
        recorded = recorded if isinstance(recorded, dict) else {}
        metrics = recorded.get("gate_metrics")
        step = recorded.get("checkpoint")
        if not isinstance(metrics, dict) or not isinstance(step, int):
            continue
        key = (
            int(bool(metrics.get("passed"))),
            -len(metrics.get("failed_gates") or []),
            float(metrics.get("balanced_accuracy") or 0.0),
            -float(metrics.get("mean_absolute_error") or 0.0),
            float(step),
        )
        if best is None or key > best[0]:
            best = (key, step)
    if best is None or best[1] not in saved:
        return None
    return best[1], saved[best[1]]


def _adapter_in(path: str) -> str | None:
    """Return the directory holding an adapter's weights, direct or one level down."""
    if _holds_adapter(path):
        return path
    try:
        children = sorted(os.listdir(path))
    except OSError:
        return None
    nested = [
        os.path.join(path, child)
        for child in children
        if os.path.isdir(os.path.join(path, child))
        and _holds_adapter(os.path.join(path, child))
    ]
    return nested[0] if len(nested) == 1 else None


def _holds_adapter(path: str) -> bool:
    """Report whether a directory holds an adapter's config and weights."""
    return os.path.isfile(os.path.join(path, ADAPTER_CONFIG)) and os.path.isfile(
        os.path.join(path, ADAPTER_WEIGHTS)
    )


def _numbered(name: str, prefix: str) -> int | None:
    """Parse ``<prefix><number>``, returning None for anything else."""
    if not name.startswith(prefix):
        return None
    try:
        return int(name.removeprefix(prefix))
    except ValueError:
        return None


def _is_run_id(name: str) -> bool:
    """Report whether a directory name has the ``YYYYMMDD-HHMMSS`` shape."""
    if len(name) != RUN_ID_LENGTH or name[8] != "-":
        return False
    return name[:8].isdigit() and name[9:].isdigit()


def _number(value: object) -> float | None:
    """Return a finite number or None for a value a run may not have written."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number


def _read_json(path: str) -> dict[str, object]:
    """Read one JSON object, treating anything unreadable as absent."""
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _first_json(path: str) -> dict[str, object]:
    """Read the metadata line of a diagnostics JSONL without loading its rows."""
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.loads(handle.readline())
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}
