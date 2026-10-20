"""Verify a recoverable interrupted release and describe what would be rebuilt.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .recovery_evidence import (
    Comparator,
    RowCounter,
    compare_adapters,
    count_training_rows,
    merge_required,
    recovered_selection,
)
from .recovery_paths import (
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    ATTEMPT_MARKER,
    TENSOR_TOLERANCE,
    RunPaths,
    adapter_digest,
    configured_paths,
    resolve_paths,
    validate_run_id,
)


@dataclass(frozen=True)
class RecoveryPlan:
    """Recovered evidence for one interrupted release, with nothing written yet."""

    paths: RunPaths
    adapter_file: str
    digest: str
    claimed_at: float
    adapter_written_at: float
    selected_step: int
    selected_path: str
    validation_metrics: dict[str, Any]
    validation_source: str
    training_rows: int
    training_rows_source: str
    merge_required: bool
    versioned_gguf: str
    notes: list[str] = field(default_factory=list)

    @property
    def claimed_test_metrics(self) -> dict[str, Any]:
        """Describe the independent test the run already scored and passed."""
        return {
            "split": "Final independent test",
            "passed": True,
            "failed_gates": [],
            "violations": [],
        }


def plan_recovery(
    run_id: str,
    *,
    root: str = ".",
    settings: RunPaths | None = None,
    digest: Callable[[str], str] | None = None,
    compare: Comparator | None = None,
    count_rows: RowCounter | None = None,
) -> RecoveryPlan:
    """Verify a recoverable interrupted release and describe what would be rebuilt."""
    run_id = validate_run_id(run_id)
    paths = resolve_paths(settings or configured_paths(run_id), root)
    digest = digest or adapter_digest
    compare = compare or compare_adapters
    count_rows = count_rows or count_training_rows
    if not os.path.isfile(os.path.join(paths.run_dir, "run_metadata.json")):
        raise SystemExit(f"[!] {paths.run_dir} is not a run directory; recovery stopped.")
    marker = os.path.join(paths.run_dir, ATTEMPT_MARKER)
    if not os.path.isdir(marker):
        raise SystemExit(
            f"[!] {paths.run_dir} never claimed its one-time independent test, so no test "
            "result can be assumed. Use the ordinary train path instead of recovery."
        )
    adapter_file = os.path.join(paths.adapter_dir, ADAPTER_WEIGHTS)
    if not os.path.isfile(adapter_file) or not os.path.isfile(
        os.path.join(paths.adapter_dir, ADAPTER_CONFIG)
    ):
        raise SystemExit(
            f"[!] {paths.adapter_dir} holds no saved selection. Recovery needs the adapter "
            "the gated release wrote before the merge."
        )
    claimed_at = os.path.getmtime(marker)
    adapter_written_at = os.path.getmtime(adapter_file)
    if adapter_written_at < claimed_at:
        raise SystemExit(
            f"[!] {adapter_file} predates the independent-test claim, so a passing test "
            "cannot be assumed; recovery stopped."
        )
    step, metrics, selected, source = recovered_selection(paths.run_dir)
    worst = compare(selected, adapter_file)
    if worst > TENSOR_TOLERANCE:
        raise SystemExit(
            f"[!] The saved adapter differs from recovered checkpoint-{step} by up to "
            f"{worst:.4f}, so it is not that selection; recovery stopped."
        )
    rows, rows_source = count_rows(paths.run_dir, paths)
    from ..deployment.provenance import versioned_gguf_name

    digest_value = digest(adapter_file)
    return RecoveryPlan(
        paths=paths,
        adapter_file=adapter_file,
        digest=digest_value,
        claimed_at=claimed_at,
        adapter_written_at=adapter_written_at,
        selected_step=step,
        selected_path=selected,
        validation_metrics=metrics,
        validation_source=source,
        training_rows=rows,
        training_rows_source=rows_source,
        merge_required=merge_required(paths.merge_dir, adapter_file, digest_value),
        versioned_gguf=versioned_gguf_name(paths.gguf, digest_value),
        notes=[f"largest saved-adapter tensor difference: {worst:.4f}"],
    )
