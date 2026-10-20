"""Compute regression and classification metrics for one scored holdout sample."""

from __future__ import annotations

import statistics
from typing import Any

from .class_metrics import class_rates, class_separation, rate_intervals
from .intervals import (
    AI_THRESHOLD,
    CONFIDENCE_LEVEL,
    _interval_record,
    apply_prevalence,
    wilson_interval,
)
from .lineage import lineage_clusters
from .metric_calculations import bootstrap_continuous_intervals, confusion

__all__ = [
    "AI_THRESHOLD",
    "CONFIDENCE_LEVEL",
    "apply_prevalence",
    "confusion",
    "lineage_clusters",
    "sample_metrics",
    "wilson_interval",
]


def sample_metrics(
    expected_scores: list[int],
    predicted_scores: list[int],
    *,
    unparsed: int = 0,
    split: str,
    classification_threshold: int = AI_THRESHOLD,
    prevalence: float | None = None,
    cluster_ids: list[str] | None = None,
    unparsed_cluster_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Measure a holdout and attach nominal 95% uncertainty intervals."""
    _validate_sample_inputs(
        expected_scores, predicted_scores, unparsed, cluster_ids, unparsed_cluster_ids
    )
    counted = confusion(
        expected_scores,
        predicted_scores,
        classification_threshold=classification_threshold,
    )
    rates = class_rates(counted)
    intervals = rate_intervals(
        expected_scores,
        classification_threshold,
        counted,
        cluster_ids,
        unparsed,
        unparsed_cluster_ids,
    )
    bootstrap = bootstrap_continuous_intervals(
        expected_scores, predicted_scores, cluster_ids
    )
    method = "source-cluster bootstrap" if cluster_ids is not None else "row bootstrap"
    intervals.update(
        {name: _interval_record(bounds, method) for name, bounds in bootstrap.items()}
    )
    total = len(expected_scores) + unparsed
    metrics = {
        "split": split,
        "examples": len(expected_scores),
        "unparsed": unparsed,
        "unparsed_rate": unparsed / total,
        "classification_threshold": classification_threshold,
        "mean_absolute_error": statistics.mean(counted["errors"]),
        "score_spread": statistics.pstdev(predicted_scores),
        "class_separation": class_separation(counted),
        **rates,
        **{
            name: counted[name]
            for name in (
                "true_positive",
                "false_positive",
                "true_negative",
                "false_negative",
            )
        },
        "ai_support": counted["true_positive"] + counted["false_negative"],
        "human_support": counted["true_negative"] + counted["false_positive"],
        "uncertainty": {
            "confidence_level": CONFIDENCE_LEVEL,
            "sampling_unit": "source cluster" if cluster_ids is not None else "row",
            "cluster_count": (
                len(set(cluster_ids + (unparsed_cluster_ids or [])))
                if cluster_ids is not None
                else total
            ),
            "note": "Intervals are nominal 95%; cluster intervals require at least ten independent lineages.",
            "metrics": intervals,
        },
    }
    if prevalence is not None:
        precision, f1 = apply_prevalence(
            metrics["recall"], metrics["specificity"], prevalence
        )
        metrics["prevalence"] = prevalence
        metrics["precision_at_prevalence"] = precision
        metrics["f1_at_prevalence"] = f1
        intervals["precision_at_prevalence"] = _interval_record(
            (0.0, 1.0), "conservative; derived metric"
        )
        intervals["f1_at_prevalence"] = _interval_record(
            (0.0, 1.0), "conservative; derived metric"
        )
    return metrics


def _validate_sample_inputs(
    expected: list[int],
    predicted: list[int],
    unparsed: int,
    clusters: list[str] | None,
    unparsed_clusters: list[str] | None,
) -> None:
    """Check sample shapes and lineage counts before calculating metrics."""
    if not expected:
        raise ValueError("sample_metrics requires at least one parsed prediction")
    if not isinstance(unparsed, int) or isinstance(unparsed, bool) or unparsed < 0:
        raise ValueError("unparsed must be a non-negative integer")
    if clusters is None and unparsed_clusters is not None:
        raise ValueError("unparsed cluster labels require parsed cluster labels")
    if clusters is not None and len(clusters) != len(expected):
        raise ValueError("cluster_ids must match the parsed holdout row count")
    if unparsed_clusters is not None and len(unparsed_clusters) != unparsed:
        raise ValueError("unparsed cluster labels must match the unparsed row count")
    if clusters is not None and unparsed and unparsed_clusters is None:
        raise ValueError(
            "unparsed source cluster labels are required when parsed clusters are used"
        )
