"""Evaluate every deterministic release gate without stopping at the first failure."""

from typing import Any

from .coverage_violations import class_support_violations, coverage_violations
from .gates import DEFAULT_SPLIT, assess_uncertainty
from .intervals import AI_THRESHOLD, CONFIDENCE_LEVEL
from .metrics import sample_metrics
from .threshold_violations import threshold_violations

_VIOLATION_ORDER = (
    "coverage",
    "unparsed",
    "class_support",
    "mean_absolute_error",
    "spread",
    "separation",
    "reliability",
)


def evaluate_holdout(
    expected_scores: list[int],
    predicted_scores: list[int],
    *,
    unparsed: int = 0,
    split: str = DEFAULT_SPLIT,
    prevalence: float | None = None,
    classification_threshold: int = AI_THRESHOLD,
    cluster_ids: list[str] | None = None,
    unparsed_cluster_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Compute regression, classification, uncertainty, and release-gate outcomes."""
    if len(expected_scores) != len(predicted_scores):
        raise ValueError("expected and predicted test scores must have equal lengths")
    total = len(expected_scores) + unparsed
    if not expected_scores:
        if not isinstance(unparsed, int) or isinstance(unparsed, bool) or unparsed < 0:
            raise ValueError("unparsed must be a non-negative integer")
        if not 0 <= classification_threshold <= 100:
            raise ValueError(
                "classification threshold must be between zero and one hundred"
            )
        if cluster_ids not in (None, []):
            raise ValueError("cluster_ids must match the parsed holdout row count")
        if cluster_ids is None and unparsed_cluster_ids is not None:
            raise ValueError("unparsed cluster labels require parsed cluster labels")
        if unparsed_cluster_ids is not None and len(unparsed_cluster_ids) != unparsed:
            raise ValueError(
                "unparsed cluster labels must match the unparsed row count"
            )
        return _unparsed_metrics(
            split,
            unparsed,
            classification_threshold,
            cluster_ids,
            unparsed_cluster_ids,
        )
    metrics = sample_metrics(
        expected_scores,
        predicted_scores,
        unparsed=unparsed,
        split=split,
        classification_threshold=classification_threshold,
        prevalence=prevalence,
        cluster_ids=cluster_ids,
        unparsed_cluster_ids=unparsed_cluster_ids,
    )
    metrics["uncertainty"]["assessment"] = assess_uncertainty(metrics)
    violations = coverage_violations(total, unparsed, split.lower())
    violations += class_support_violations(metrics, split.lower())
    violations += threshold_violations(metrics)
    violations.sort(key=lambda item: _VIOLATION_ORDER.index(item["gate"]))
    metrics["violations"] = violations
    metrics["failed_gates"] = [item["gate"] for item in violations]
    metrics["passed"] = not violations
    return metrics


def _unparsed_metrics(
    split: str,
    unparsed: int,
    threshold: int,
    clusters: list[str] | None,
    unparsed_clusters: list[str] | None,
) -> dict[str, Any]:
    """Represent a wholly unparseable holdout as explicit, fail-closed gate metrics."""
    total = unparsed
    intervals = {
        name: {
            "lower": lower,
            "upper": upper,
            "method": "no parseable predictions; full-domain interval",
        }
        for name, lower, upper in (
            ("mean_absolute_error", 0.0, 100.0),
            ("score_spread", 0.0, 50.0),
            ("class_separation", -100.0, 100.0),
            ("precision", 0.0, 1.0),
            ("recall", 0.0, 1.0),
            ("specificity", 0.0, 1.0),
            ("unparsed_rate", 0.0, 1.0),
        )
    }
    parsed_clusters = clusters or []
    all_clusters = parsed_clusters + (unparsed_clusters or [])
    metrics = {
        "split": split,
        "examples": 0,
        "unparsed": unparsed,
        "unparsed_rate": unparsed / total if total else 1.0,
        "classification_threshold": threshold,
        "mean_absolute_error": 100.0,
        "score_spread": 0.0,
        "class_separation": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "specificity": 0.0,
        "f1": 0.0,
        "balanced_accuracy": 0.0,
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
        "ai_support": 0,
        "human_support": 0,
        "scoring_status": "NO_PARSEABLE_PREDICTIONS",
        "uncertainty": {
            "confidence_level": CONFIDENCE_LEVEL,
            "sampling_unit": "source cluster" if clusters is not None else "row",
            "cluster_count": (
                len(set(all_clusters)) if clusters is not None else total
            ),
            "note": "No predictions were parseable; metric values are conservative failure sentinels.",
            "metrics": intervals,
        },
    }
    metrics["uncertainty"]["assessment"] = assess_uncertainty(metrics)
    violations = coverage_violations(total, unparsed, split.lower())
    violations += class_support_violations(metrics, split.lower())
    violations += threshold_violations(metrics)
    violations.sort(key=lambda item: _VIOLATION_ORDER.index(item["gate"]))
    metrics["violations"] = violations
    metrics["failed_gates"] = [item["gate"] for item in violations]
    metrics["passed"] = False
    return metrics
