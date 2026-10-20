"""Calculate per-class rates and their clustered uncertainty intervals."""

import statistics
from typing import Any

from .intervals import AI_THRESHOLD, _cluster_adjusted_interval, _interval_record
from .lineage import group_counts


def class_rates(counted: dict[str, Any]) -> dict[str, float]:
    """Calculate precision, recall, specificity, F1, and balanced accuracy."""
    ai = counted["true_positive"] + counted["false_negative"]
    human = counted["true_negative"] + counted["false_positive"]
    predicted = counted["true_positive"] + counted["false_positive"]
    precision = counted["true_positive"] / predicted if predicted else 0.0
    recall = counted["true_positive"] / ai if ai else 0.0
    specificity = counted["true_negative"] / human if human else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def rate_intervals(
    expected: list[int],
    threshold: int,
    counted: dict[str, Any],
    clusters: list[str] | None,
    unparsed: int,
    unparsed_clusters: list[str] | None,
) -> dict[str, dict[str, Any]]:
    """Build class-conditional and parse-coverage intervals."""
    expected_ai = [score >= AI_THRESHOLD for score in expected]
    predicted_ai = [score >= threshold for score in counted["predicted_scores"]]
    true_positive = [
        actual and predicted
        for actual, predicted in zip(expected_ai, predicted_ai, strict=True)
    ]
    true_negative = [
        not actual and not predicted
        for actual, predicted in zip(expected_ai, predicted_ai, strict=True)
    ]
    counts = {
        "precision": group_counts(predicted_ai, true_positive, clusters),
        "recall": group_counts(expected_ai, true_positive, clusters),
        "specificity": group_counts(
            [not value for value in expected_ai], true_negative, clusters
        ),
    }
    totals = {
        "precision": (
            counted["true_positive"],
            counted["true_positive"] + counted["false_positive"],
        ),
        "recall": (
            counted["true_positive"],
            counted["true_positive"] + counted["false_negative"],
        ),
        "specificity": (
            counted["true_negative"],
            counted["true_negative"] + counted["false_positive"],
        ),
    }
    intervals = {}
    for name in totals:
        interval, method = _cluster_adjusted_interval(*totals[name], counts[name])
        intervals[name] = _interval_record(interval, method)
    total = len(expected) + unparsed
    all_clusters = (
        clusters + (unparsed_clusters or []) if clusters is not None else None
    )
    unparsed_counts = group_counts(
        [True] * total, [False] * len(expected) + [True] * unparsed, all_clusters
    )
    interval, method = _cluster_adjusted_interval(unparsed, total, unparsed_counts)
    intervals["unparsed_rate"] = _interval_record(interval, method)
    return intervals


def class_separation(counted: dict[str, Any]) -> float:
    """Measure predicted-score difference between the two target classes."""
    ai = counted["ai_predictions"]
    human = counted["human_predictions"]
    return statistics.mean(ai) - statistics.mean(human) if ai and human else 0.0
