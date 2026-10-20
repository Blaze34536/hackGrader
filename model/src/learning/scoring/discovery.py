"""Find saved checkpoints and load the holdout splits used to score them.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from ..config import COMBINED_FILE, RUN_DIR, TEST_FILE, VAL_FILE
from ..preflight import load_rows, partition_combined_rows

ROUND_PATTERN = re.compile(r"^round-(\d+)$")
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def checkpoint_directory(adapter_path: str) -> str:
    """Return the Trainer checkpoint directory containing an adapter artifact."""
    path = os.path.normpath(adapter_path)
    base = os.path.basename(path)
    if CHECKPOINT_PATTERN.fullmatch(base):
        return path
    parent = os.path.dirname(path)
    if CHECKPOINT_PATTERN.fullmatch(os.path.basename(parent)):
        return parent
    if base.startswith(("continuation", "gate-", "resume-score-")):
        grandparent = os.path.dirname(parent)
        if CHECKPOINT_PATTERN.fullmatch(os.path.basename(grandparent)):
            return grandparent
    return path


def checkpoint_run_dir(adapter_path: str) -> str:
    """Return the run directory containing a direct or round-local checkpoint."""
    checkpoint = checkpoint_directory(adapter_path)
    parent = os.path.dirname(checkpoint)
    if ROUND_PATTERN.fullmatch(os.path.basename(parent)):
        return os.path.dirname(parent)
    return parent


def adapter_directory(checkpoint_path: str) -> str | None:
    """Find a direct adapter or its single nested PEFT adapter directory."""
    if os.path.isfile(os.path.join(checkpoint_path, "adapter_config.json")):
        return checkpoint_path
    if any(
        os.path.isfile(os.path.join(checkpoint_path, name))
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        return checkpoint_path
    try:
        children = os.listdir(checkpoint_path)
    except (NotADirectoryError, FileNotFoundError):
        return None
    adapters = [
        os.path.join(checkpoint_path, child)
        for child in children
        if os.path.isdir(os.path.join(checkpoint_path, child))
        and (
            os.path.isfile(os.path.join(checkpoint_path, child, "adapter_config.json"))
            or any(
                os.path.isfile(os.path.join(checkpoint_path, child, name))
                for name in ("adapter_model.safetensors", "adapter_model.bin")
            )
        )
    ]
    if len(adapters) > 1:
        raise SystemExit(
            f"[!] Multiple adapter directories found under {checkpoint_path!r}; "
            "checkpoint discovery stopped."
        )
    return adapters[0] if adapters else None


def checkpoint_steps(run_dir: str) -> list[tuple[int, str]]:
    """List saved adapter checkpoints in a run directory, oldest logical step first.

    Numbered training rounds each restart their optimizer-step counter, so the run
    metadata reserves a disjoint logical-step range for every round. Legacy runs with
    checkpoint directories directly under ``run_dir`` remain readable.
    """
    try:
        entries = os.listdir(run_dir)
    except FileNotFoundError:
        raise SystemExit(f"[!] Run directory {run_dir!r} does not exist.") from None

    metadata_path = os.path.join(run_dir, "run_metadata.json")
    stride = None
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            stride = int(json.load(handle).get("max_steps_per_round", 0))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass

    found: list[tuple[int, str]] = []
    logical_steps: set[int] = set()
    for entry in entries:
        match = CHECKPOINT_PATTERN.match(entry)
        if match:
            checkpoint_path = os.path.join(run_dir, entry)
            adapter_path = (
                adapter_directory(checkpoint_path)
                if os.path.isdir(checkpoint_path)
                else None
            )
            if adapter_path is not None:
                step = int(match.group(1))
                found.append((step, adapter_path))
                logical_steps.add(step)
            continue

        round_match = ROUND_PATTERN.match(entry)
        if not round_match or stride is None or stride < 1:
            continue
        round_number = int(round_match.group(1))
        round_dir = os.path.join(run_dir, entry)
        if not os.path.isdir(round_dir):
            continue
        for checkpoint in os.listdir(round_dir):
            checkpoint_match = CHECKPOINT_PATTERN.match(checkpoint)
            if not checkpoint_match:
                continue
            checkpoint_path = os.path.join(round_dir, checkpoint)
            adapter_path = (
                adapter_directory(checkpoint_path)
                if os.path.isdir(checkpoint_path)
                else None
            )
            if adapter_path is None:
                continue
            local_step = int(checkpoint_match.group(1))
            logical_step = (round_number - 1) * stride + local_step
            if logical_step in logical_steps:
                raise SystemExit(
                    f"[!] Duplicate logical checkpoint step {logical_step} under {run_dir!r}."
                )
            logical_steps.add(logical_step)
            found.append((logical_step, adapter_path))
    return sorted(found)


def outputs_root() -> str:
    """Return the directory that holds run subdirectories."""
    return os.path.dirname(RUN_DIR) or "outputs"


def newest_run_dir() -> str:
    """Return the most recently modified run directory that holds checkpoints."""
    root = outputs_root()
    if not os.path.isdir(root):
        raise SystemExit(
            f"[!] No {root!r} directory with saved checkpoints was found; pass --run-dir."
        )
    candidates = sorted(
        (os.path.join(root, name) for name in os.listdir(root)),
        key=os.path.getmtime,
        reverse=True,
    )
    for candidate in candidates:
        if os.path.isdir(candidate) and checkpoint_steps(candidate):
            return candidate
    raise SystemExit(
        f"[!] No run directory under {root!r} holds a checkpoint-* adapter; pass "
        "--run-dir."
    )


def load_scoring_splits(
    *,
    dataset_file: str | None = None,
    validation_file: str | None = None,
    test_file: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the validation and independent-test partitions used for scoring."""
    import contracts as report

    if dataset_file is not None and any((validation_file, test_file)):
        raise SystemExit(
            "[!] Use either --dataset or --validation-file/--test-file, not both."
        )
    if dataset_file is None and not any((validation_file, test_file)):
        dataset_file = COMBINED_FILE
    if dataset_file is not None:
        if not os.path.exists(dataset_file):
            raise SystemExit(f"[!] Missing combined dataset {dataset_file!r}.")
        rows = load_rows(dataset_file, report)
        _training, validation_rows, test_rows = partition_combined_rows(
            rows, dataset_file
        )
        return validation_rows, test_rows
    return (
        load_rows(validation_file or VAL_FILE, report),
        load_rows(test_file or TEST_FILE, report),
    )
