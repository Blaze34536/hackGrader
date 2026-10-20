"""Report holdout gate results and summarize their uncertainty intervals."""

from __future__ import annotations

from typing import Any

from ..config import (
    HOLDOUT_MAE_LIMIT,
    HOLDOUT_MIN_PRECISION,
    HOLDOUT_MIN_RECALL,
    HOLDOUT_MIN_SPECIFICITY,
    MIN_HOLDOUT_SEPARATION,
    MIN_HOLDOUT_SPREAD,
)
from .intervals import AI_THRESHOLD

DEFAULT_SPLIT = "Final independent test"
UNPARSEABLE_RATE_LIMIT = 0.05


def assess_uncertainty(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Compare nominal intervals with release floors without waiving deterministic gates."""
    limits = {
        "mean_absolute_error": ("maximum", HOLDOUT_MAE_LIMIT),
        "score_spread": ("minimum", MIN_HOLDOUT_SPREAD),
        "class_separation": ("minimum", MIN_HOLDOUT_SEPARATION),
        "precision": ("minimum", HOLDOUT_MIN_PRECISION),
        "recall": ("minimum", HOLDOUT_MIN_RECALL),
        "specificity": ("minimum", HOLDOUT_MIN_SPECIFICITY),
        "unparsed_rate": ("maximum", UNPARSEABLE_RATE_LIMIT),
    }
    intervals = metrics["uncertainty"]["metrics"]
    assessment = {
        name: {
            **interval,
            "direction": direction,
            "threshold": threshold,
            "status": _uncertainty_status(interval, threshold, direction=direction),
        }
        for name, (direction, threshold) in limits.items()
        if (interval := intervals.get(name)) is not None
    }
    assessment.update(
        {
            name: intervals[name]
            for name in ("precision_at_prevalence", "f1_at_prevalence")
            if name in intervals
        }
    )
    return assessment


def _uncertainty_status(
    interval: dict[str, Any], threshold: float, *, direction: str
) -> str:
    """Classify interval position without letting uncertainty waive a gate."""
    lower, upper = interval["lower"], interval["upper"]
    if direction == "minimum":
        if lower >= threshold:
            return "PASS"
        if upper < threshold:
            return "FAIL"
    elif direction == "maximum":
        if upper <= threshold:
            return "PASS"
        if lower > threshold:
            return "FAIL"
    else:
        raise ValueError(f"unknown uncertainty direction {direction!r}")
    return "INCONCLUSIVE"


def summarize_holdout(metrics: dict[str, Any]) -> str:
    """Render holdout metrics and nominal uncertainty status."""
    threshold = metrics.get("classification_threshold", AI_THRESHOLD)
    uncertainty = metrics.get("uncertainty", {})
    assessed = uncertainty.get("assessment", {})
    details = []
    for name, interval_data in uncertainty.get("metrics", {}).items():
        interval = assessed.get(name) or interval_data
        if interval:
            unit = " points" if name == "class_separation" else ""
            details.append(
                f"{name} [{interval['lower']:.2f}, {interval['upper']:.2f}]"
                f"{unit} {interval.get('status', '')}".rstrip()
            )
    summary = (
        f"[i] {metrics['split']}: {metrics['examples']} scored / {metrics['unparsed']} "
        f"unparsed; cutoff {threshold}; MAE {metrics['mean_absolute_error']:.1f}; "
        f"precision {metrics['precision']:.2f}; recall {metrics['recall']:.2f}; "
        f"specificity {metrics['specificity']:.2f}; F1 {metrics['f1']:.2f}; "
        f"class separation {metrics['class_separation']:.1f} points."
    )
    failures = metrics.get("format_failures") or {}
    if failures:
        summary += (
            " Format diagnostics (count only, no generated text): "
            + ", ".join(f"{name}={count}" for name, count in sorted(failures.items()))
            + "."
        )
    if details:
        unit = uncertainty.get("sampling_unit", "row")
        count = uncertainty.get("cluster_count")
        if count is not None:
            unit += f" ({count} independent source clusters)"
        summary += (
            f" Nominal 95% {unit} intervals: {', '.join(details)}. "
            "INCONCLUSIVE overlaps its gate floor and does not waive the gate."
        )
    return summary


def report_release_gates(label: str, metrics: dict[str, Any]) -> None:
    """Print one holdout's outcome, listing every failed gate."""
    print(
        f"[i] {label} confusion: {metrics['true_positive']} AI found, "
        f"{metrics['false_negative']} AI missed, {metrics['false_positive']} false "
        f"alarms, {metrics['true_negative']} human confirmed."
    )
    if metrics["passed"]:
        print(f"[i] {label}: every release gate passed.")
        return
    print(f"[!] {label}: {len(metrics['failed_gates'])} release gate(s) failed.")
    for failure in format_gate_failures(metrics):
        print(f"    {failure}")


def format_gate_failures(metrics: dict[str, Any]) -> list[str]:
    """Number every gate failure and include its measured value and limit."""
    violations = metrics["violations"]
    total = len(violations)
    return [
        f"[{index}/{total}] {item['gate']}: {item['message'].removeprefix('[!] ')}"
        for index, item in enumerate(violations, start=1)
    ]
