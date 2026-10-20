"""Score one saved checkpoint and rank checkpoints against each other.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..holdout import (
    AI_THRESHOLD,
    evaluate_holdout,
    generate_holdout_predictions,
    lineage_clusters,
    require_test_coverage,
)
from ..inputs import ai_prevalence
from ..pipeline.models import load_selected_model
from ..training.artifacts import require_time_remaining

VALIDATION_LABEL = "Validation holdout"
TEST_LABEL = "Final independent test"
SPLIT_LABELS = {"validation": VALIDATION_LABEL, "test": TEST_LABEL}


def _prediction_clusters(
    predictions: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Assign parsed and unparsed rows cluster IDs from the same full prediction set."""
    all_clusters = lineage_clusters(predictions)
    parsed_clusters = [
        cluster
        for record, cluster in zip(predictions, all_clusters, strict=True)
        if record["parsed"]
    ]
    unparsed_clusters = [
        cluster
        for record, cluster in zip(predictions, all_clusters, strict=True)
        if not record["parsed"]
    ]
    return parsed_clusters, unparsed_clusters


def evaluate_split(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    limit: int,
    *,
    split: str,
    classification_threshold: int = AI_THRESHOLD,
    batch_size: int = 1,
    predictions: list[dict[str, Any]] | None = None,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Score one checkpoint against a holdout split without enforcing any gate."""
    require_time_remaining(deadline, "holdout scoring")
    selected = require_test_coverage(rows, limit)
    if predictions is None:
        predictions = generate_holdout_predictions(
            model,
            tokenizer,
            selected,
            classification_threshold=classification_threshold,
            batch_size=batch_size,
            deadline=deadline,
        )
    elif len(predictions) != len(selected):
        raise ValueError("prediction row count does not match the selected holdout")

    parsed = [record for record in predictions if record["parsed"]]
    expected_scores = [record["target_score"] for record in parsed]
    predicted_scores = [record["predicted_score"] for record in parsed]
    unparsed = len(predictions) - len(parsed)
    parsed_clusters, unparsed_clusters = _prediction_clusters(predictions)
    require_time_remaining(deadline, "holdout scoring")
    metrics = evaluate_holdout(
        expected_scores,
        predicted_scores,
        unparsed=unparsed,
        split=split,
        prevalence=ai_prevalence(rows),
        classification_threshold=classification_threshold,
        cluster_ids=parsed_clusters,
        unparsed_cluster_ids=unparsed_clusters,
    )
    metrics["predictions"] = predictions
    format_failures = Counter(
        failure
        for record in predictions
        for failure in record.get("format_failures", [])
    )
    metrics["format_failures"] = dict(sorted(format_failures.items()))
    return metrics


def score_checkpoint(
    dependencies: dict[str, Any],
    checkpoint_path: str,
    splits: dict[str, tuple[list[dict[str, Any]], str]],
    limit: int,
    *,
    classification_threshold: int = AI_THRESHOLD,
    batch_size: int = 1,
    deadline: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Load one saved adapter over its base model and score every requested split."""
    model, tokenizer = load_selected_model(dependencies, checkpoint_path)
    results: dict[str, dict[str, Any]] = {}
    try:
        for name, (rows, label) in splits.items():
            results[name] = evaluate_split(
                model,
                tokenizer,
                rows,
                limit,
                split=label,
                classification_threshold=classification_threshold,
                batch_size=batch_size,
                deadline=deadline,
            )
    finally:
        del model
        dependencies["torch"].cuda.empty_cache()
    return results


def sweep_thresholds(
    predictions: list[dict[str, Any]],
    thresholds: list[int],
    *,
    split: str,
    prevalence: float | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """Evaluate saved predictions at several cutoffs without repeating model inference."""
    if not thresholds:
        raise ValueError("at least one classification threshold is required")
    if any(not 0 <= threshold <= 100 for threshold in thresholds):
        raise ValueError(
            "classification thresholds must be between zero and one hundred"
        )
    parsed = [record for record in predictions if record["parsed"]]
    expected = [record["target_score"] for record in parsed]
    predicted = [record["predicted_score"] for record in parsed]
    if not expected:
        raise ValueError(
            "no parseable predictions are available for threshold calibration"
        )
    unparsed = len(predictions) - len(parsed)
    clusters, unparsed_clusters = _prediction_clusters(predictions)
    records = []
    for threshold in thresholds:
        metrics = evaluate_holdout(
            expected,
            predicted,
            unparsed=unparsed,
            split=split,
            prevalence=prevalence,
            classification_threshold=threshold,
            cluster_ids=clusters,
            unparsed_cluster_ids=unparsed_clusters,
        )
        records.append((threshold, metrics))
    return records


def rank_key(step: int, metrics: dict[str, Any]) -> tuple[int, float, float, int]:
    """Rank checkpoints: clearing every gate first, then separation quality, then step."""
    return (
        int(bool(metrics["passed"])),
        metrics["balanced_accuracy"],
        -metrics["mean_absolute_error"],
        step,
    )


def best_checkpoint(records: list[tuple[int, str, dict[str, Any]]]) -> int | None:
    """Return the checkpoint with the best holdout behaviour across the scored splits."""
    if not records:
        return None
    return max(records, key=lambda item: rank_key(item[0], item[2]))[0]
