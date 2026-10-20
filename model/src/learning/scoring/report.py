"""Render the checkpoint score table and the sweep's recommendation.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import collections
from typing import Any

from .measure import best_checkpoint

GATE_SHORT_NAMES = {
    "coverage": "coverage",
    "unparsed": "unparsed",
    "class_support": "class-support",
    "mean_absolute_error": "mae",
    "spread": "spread",
    "separation": "separation",
    "reliability": "reliability",
}


def format_score_table(records: list[tuple[int, str, dict[str, Any]]]) -> str:
    """Render one aligned row per checkpoint and holdout split."""
    header = (
        f"{'step':>5}  {'split':<22} {'MAE':>6} {'prec':>5} {'rec':>5} "
        f"{'spec':>5} {'sep':>6} {'bal':>5}  gates"
    )
    lines = [header, "-" * len(header)]
    for step, split_name, metrics in records:
        gates = ",".join(
            GATE_SHORT_NAMES.get(gate, gate) for gate in metrics["failed_gates"]
        )
        lines.append(
            f"{step:>5}  {split_name:<22} {metrics['mean_absolute_error']:>6.1f} "
            f"{metrics['precision']:>5.2f} {metrics['recall']:>5.2f} "
            f"{metrics['specificity']:>5.2f} {metrics['class_separation']:>6.1f} "
            f"{metrics['balanced_accuracy']:>5.2f}  {gates or 'pass'}"
        )
    return "\n".join(lines)


def pass_by_step(records: list[tuple[int, str, dict[str, Any]]]) -> dict[int, bool]:
    """Report whether each checkpoint cleared every gate on every scored split."""
    grouped: dict[int, list[bool]] = collections.defaultdict(list)
    for step, _split_name, metrics in records:
        grouped[step].append(bool(metrics["passed"]))
    return {step: all(results) for step, results in grouped.items()}


def report_sweep(
    records: list[tuple[int, str, dict[str, Any]]],
    *,
    stream=print,
) -> bool:
    """Print the score table, the failing-gate tally, and the recommended checkpoint."""
    if not records:
        stream("[!] No checkpoint produced holdout metrics.")
        return False
    stream("")
    for _step, _split_name, metrics in records:
        prevalence = metrics.get("prevalence")
        if prevalence is None:
            continue
        stream(
            f"[i] {metrics['split']} at {prevalence:.0%} machine-authored prevalence: "
            f"precision {metrics['precision_at_prevalence']:.2f}; "
            f"F1 {metrics['f1_at_prevalence']:.2f}"
        )
    stream("")
    stream(format_score_table(records))
    stream("")
    failures = collections.Counter(
        gate
        for _step, _split_name, metrics in records
        for gate in metrics["failed_gates"]
    )
    cleared = [step for step, passed in pass_by_step(records).items() if passed]
    if failures:
        stream(
            "[i] failing gates: "
            + ", ".join(f"{name} x{count}" for name, count in failures.most_common())
        )
    if cleared:
        recommended = best_checkpoint(records)
        stream(f"[i] checkpoint(s) clearing every scored gate: {sorted(cleared)}")
        stream(
            f"[i] recommended checkpoint: checkpoint-{recommended}. Review the table "
            "above before merging or deploying it."
        )
        return True
    stream(
        "[!] No scored checkpoint clears every gate. Use the failed metrics and "
        "validation evidence to diagnose whether data, training, or gate settings need "
        "attention."
    )
    return False
