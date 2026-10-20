"""Render a recovery plan and record how a rebuilt release was produced.

The stamp written beside merged weights carries the release gates a normal merge would
have recorded. The recovery record next to it carries what the stamp cannot: that the
claim survived an interrupted merge and that the one-time test's gate metrics were never
persisted, because the pipeline only writes them after a completed merge.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from .recovery_paths import ATTEMPT_MARKER, RECOVERY_RECORD
from .recovery_plan import RecoveryPlan


def _stamp_time(value: float) -> str:
    """Format one filesystem timestamp for reports and provenance records."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(value))


def report_plan(plan: RecoveryPlan) -> None:
    """Print the recovered evidence before anything is written."""
    print(f"[i] recovering run {plan.paths.run_id} from {plan.paths.run_dir}")
    print(
        f"[i] independent test claimed {_stamp_time(plan.claimed_at)}; selected adapter "
        f"written {_stamp_time(plan.adapter_written_at)}, so that test passed."
    )
    print(
        f"[i] recovered selection: checkpoint-{plan.selected_step} "
        f"({plan.selected_path}) from {plan.validation_source}"
    )
    print(f"[i] saved adapter: {plan.adapter_file} (sha256 {plan.digest})")
    print(
        f"[i] training rows recorded as {plan.training_rows} from "
        f"{plan.training_rows_source}"
    )
    for note in plan.notes:
        print(f"[i] {note}")
    print(
        "[i] this recovery re-scores nothing: the reserve independent test is not "
        "evaluated again."
    )


def report_planned_commands(plan: RecoveryPlan) -> None:
    """Print the merge, conversion, and registration this recovery would run."""
    from ..deployment.provenance import CONVERTER, GGUF_TYPE, MODELFILE

    if plan.merge_required:
        print(f"[i] planned merge: 16-bit weights into {plan.paths.merge_dir}")
    else:
        print(
            "[i] merged weights already carry this adapter's stamp; only conversion and "
            "registration are planned."
        )
    print(
        "[i] planned command: "
        + " ".join(
            [
                sys.executable,
                CONVERTER,
                plan.paths.merge_dir,
                "--outfile",
                plan.versioned_gguf,
                "--outtype",
                GGUF_TYPE,
            ]
        )
    )
    print(
        f"[i] planned command: ollama create {plan.paths.tag}:{plan.digest[:16]} "
        f"-f <generated {MODELFILE}>"
    )
    print("[i] dry run: nothing was merged, converted, or registered.")


def evidence(plan: RecoveryPlan) -> dict[str, Any]:
    """Build the provenance record that explains how this build was recovered."""
    from . import release

    return {
        "run_id": plan.paths.run_id,
        "recovered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "adapter": {
            "path": plan.adapter_file,
            "sha256": plan.digest,
            "written_at": _stamp_time(plan.adapter_written_at),
        },
        "selected_checkpoint": {
            "step": plan.selected_step,
            "path": plan.selected_path,
            "comparison": plan.notes,
        },
        "validation_metrics": release._release_metric_summary(plan.validation_metrics),
        "validation_metrics_source": plan.validation_source,
        "training_rows": plan.training_rows,
        "training_rows_source": plan.training_rows_source,
        "independent_test": {
            "attempt_marker": ATTEMPT_MARKER,
            "claimed_at": _stamp_time(plan.claimed_at),
            "metrics_archived": False,
            "note": (
                "The one-time independent test was claimed and passed before the "
                "interrupted merge. The pipeline records its gate metrics only after a "
                "completed merge, so they were never persisted, and the reserve test was "
                "not scored again during recovery."
            ),
        },
    }


def record_evidence(plan: RecoveryPlan) -> str:
    """Write the recovery provenance beside the merged weights."""
    os.makedirs(plan.paths.merge_dir, exist_ok=True)
    path = os.path.join(plan.paths.merge_dir, RECOVERY_RECORD)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(evidence(plan), handle, indent=2)
    print(f"[i] recovery provenance written to {path}.")
    return path



