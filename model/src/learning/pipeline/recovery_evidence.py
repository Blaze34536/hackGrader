"""Recover the selected checkpoint and prove the saved adapter still matches it.

A recovery may only rebuild the merge for the checkpoint the interrupted run actually
selected. The selection is re-derived from the run's own validation diagnostics, and the
saved weights are then compared to that checkpoint tensor by tensor, so a mismatch stops
the rebuild instead of merging the wrong adapter.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import glob
import json
import os
from collections.abc import Callable
from typing import Any

from .recovery_paths import (
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    DIAGNOSTICS_PREFIX,
    STAMP_NAME,
    RunPaths,
)

RowCounter = Callable[[str, RunPaths], tuple[int, str]]
Comparator = Callable[[str, str], float]


def weights_file(path: str) -> str:
    """Return the weight file inside a saved adapter or checkpoint directory."""
    return os.path.join(path, ADAPTER_WEIGHTS) if os.path.isdir(path) else path


def compare_adapters(selected_adapter: str, saved_adapter: str) -> float:
    """Return the largest absolute weight difference between two saved LoRA adapters."""
    selected_adapter = weights_file(selected_adapter)
    saved_adapter = weights_file(saved_adapter)
    try:
        import torch
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit(
            f"[!] Adapter verification needs safetensors and torch ({exc}); recovery "
            "stopped."
        ) from None
    try:
        with (
            safe_open(selected_adapter, framework="pt") as selected,
            safe_open(saved_adapter, framework="pt") as saved,
        ):
            selected_keys = set(selected.keys())
            saved_keys = set(saved.keys())
            if selected_keys != saved_keys:
                raise SystemExit(
                    "[!] The saved adapter and the recovered checkpoint hold "
                    f"{len(selected_keys ^ saved_keys)} differing tensor name(s); "
                    "recovery stopped."
                )
            worst = 0.0
            for key in sorted(selected_keys):
                left = selected.get_tensor(key).to(torch.float32)
                right = saved.get_tensor(key).to(torch.float32)
                if left.shape != right.shape:
                    raise SystemExit(
                        f"[!] Tensor {key!r} differs in shape between the saved adapter "
                        "and the recovered checkpoint; recovery stopped."
                    )
                worst = max(worst, float((left - right).abs().max()))
    except SystemExit:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(
            f"[!] Saved adapters could not be compared ({type(exc).__name__}: {exc}); "
            "recovery stopped."
        ) from None
    return worst


def selection_key(candidate: tuple[int, dict[str, Any], str, str]) -> tuple[Any, ...]:
    """Rank recovered diagnostics exactly as the in-memory validation sweep does."""
    step, metrics, _path, _source = candidate
    return (
        int(bool(metrics.get("passed"))),
        -len(metrics.get("failed_gates") or []),
        metrics.get("balanced_accuracy") or 0.0,
        -(metrics.get("mean_absolute_error") or 0.0),
        step,
    )


def recovered_selection(run_dir: str) -> tuple[int, dict[str, Any], str, str]:
    """Recover the checkpoint the interrupted run selected from its gate diagnostics."""
    candidates: list[tuple[int, dict[str, Any], str, str]] = []
    for name in sorted(os.listdir(run_dir)):
        if not name.startswith(DIAGNOSTICS_PREFIX) or not name.endswith(".jsonl"):
            continue
        source = os.path.join(run_dir, name)
        try:
            with open(source, encoding="utf-8") as handle:
                metadata = json.loads(handle.readline()).get("metadata")
        except (OSError, AttributeError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"[!] Cannot read validation diagnostics {source} "
                f"({type(exc).__name__}: {exc}); recovery stopped."
            ) from None
        recorded = metadata if isinstance(metadata, dict) else {}
        metrics = recorded.get("gate_metrics")
        step = recorded.get("checkpoint")
        checkpoint_path = recorded.get("checkpoint_path")
        if not isinstance(metrics, dict) or not isinstance(step, int):
            raise SystemExit(
                f"[!] {source} records no checkpoint step and gate metrics; "
                "recovery stopped."
            )
        if not checkpoint_path:
            raise SystemExit(f"[!] {source} records no checkpoint path; recovery stopped.")
        candidates.append((step, metrics, os.path.normpath(checkpoint_path), source))
    if not candidates:
        raise SystemExit(
            f"[!] {run_dir} holds no {DIAGNOSTICS_PREFIX}*.jsonl validation diagnostics, "
            "so the selected checkpoint cannot be recovered; recovery stopped."
        )
    step, metrics, checkpoint_path, source = max(candidates, key=selection_key)
    from ..scoring.discovery import adapter_directory

    adapter = adapter_directory(checkpoint_path)
    if adapter is None or not os.path.isfile(os.path.join(adapter, ADAPTER_CONFIG)):
        raise SystemExit(
            f"[!] The recovered selection {checkpoint_path} holds no saved adapter; "
            "recovery stopped."
        )
    return step, metrics, adapter, source


def merge_required(merge_dir: str, adapter_file: str, digest: str) -> bool:
    """Report whether a completed merge of this exact adapter is missing.

    A partial merge also leaves fresh shards behind, so shards alone cannot prove the
    weights are whole. Only the stamp, which is written after the merge is verified and
    names the adapter it came from, allows a recovery to skip merging again.

    ``weight_time`` is imported here rather than at module level because deployment's
    provenance reads the model configuration, and the configuration resolves the run id
    when it is first imported. The recovery entry point pins that id before planning.
    """
    from ..deployment.provenance import weight_time

    try:
        with open(os.path.join(merge_dir, STAMP_NAME), encoding="utf-8") as handle:
            recorded = json.load(handle).get("adapter_sha256")
    except (OSError, AttributeError, json.JSONDecodeError):
        return True
    if recorded != digest:
        return True
    shards = sorted(glob.glob(os.path.join(merge_dir, "*.safetensors")))
    if not shards:
        return True
    adapter_time = weight_time(adapter_file)
    return any(weight_time(shard) < adapter_time for shard in shards)


def count_training_rows(run_dir: str, paths: RunPaths) -> tuple[int, str]:
    """Count the combined dataset's training rows that this run started from."""
    from .recovery_paths import run_started_at

    if not os.path.isfile(paths.dataset):
        raise SystemExit(
            f"[!] The run's combined dataset {paths.dataset!r} is missing, so the "
            "training-row count cannot be recorded; recovery stopped."
        )
    if os.path.getmtime(paths.dataset) > run_started_at(run_dir):
        raise SystemExit(
            f"[!] {paths.dataset} was modified after the run started, so it may not be "
            "the data this adapter was trained on; recovery stopped."
        )
    import contracts as report

    from ..preflight import load_rows, partition_combined_rows

    train_rows, _validation, _test = partition_combined_rows(
        load_rows(paths.dataset, report), paths.dataset
    )
    return len(train_rows) * paths.repeat, f"{paths.dataset} train split x{paths.repeat}"
