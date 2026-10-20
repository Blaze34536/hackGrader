"""Compute continuous uncertainty samples and prediction confusion counts."""

import random
import statistics
from collections import defaultdict

from .intervals import AI_THRESHOLD, _percentile
from .lineage import cluster_values

_BOOTSTRAP_SAMPLES = 1000
_MIN_INDEPENDENT_CLUSTERS = 10


def bootstrap_continuous_intervals(
    expected_scores: list[int],
    predicted_scores: list[int],
    cluster_ids: list[str] | None,
) -> dict[str, tuple[float, float]]:
    """Estimate MAE, score spread, and separation by resampling source lineage."""
    domains = {
        "mean_absolute_error": (0.0, 100.0),
        "score_spread": (0.0, 50.0),
        "class_separation": (-100.0, 100.0),
    }
    count = len(expected_scores)
    if cluster_ids is not None and len(cluster_ids) != count:
        raise ValueError("cluster_ids must match the parsed holdout row count")
    independent_count = len(set(cluster_ids)) if cluster_ids is not None else count
    minimum = _MIN_INDEPENDENT_CLUSTERS if cluster_ids is not None else 2
    if independent_count < minimum:
        return domains
    clusters = cluster_values([0.0] * count, cluster_ids)
    rng = random.Random(42)
    estimates: dict[str, list[float]] = defaultdict(list)
    for _ in range(_BOOTSTRAP_SAMPLES):
        indices = [
            index for _ in clusters for index in clusters[rng.randrange(len(clusters))]
        ]
        sampled_scores = [predicted_scores[index] for index in indices]
        errors = [
            abs(predicted_scores[index] - expected_scores[index]) for index in indices
        ]
        ai_scores = [
            predicted_scores[index]
            for index in indices
            if expected_scores[index] >= AI_THRESHOLD
        ]
        human_scores = [
            predicted_scores[index]
            for index in indices
            if expected_scores[index] < AI_THRESHOLD
        ]
        estimates["mean_absolute_error"].append(statistics.mean(errors))
        estimates["score_spread"].append(statistics.pstdev(sampled_scores))
        if ai_scores and human_scores:
            estimates["class_separation"].append(
                statistics.mean(ai_scores) - statistics.mean(human_scores)
            )
    output = {}
    for name, bounds in domains.items():
        sample = estimates[name]
        if len(sample) < _BOOTSTRAP_SAMPLES * 0.95:
            output[name] = bounds
        else:
            sample.sort()
            output[name] = (_percentile(sample, 0.025), _percentile(sample, 0.975))
    return output


def confusion(
    expected_scores: list[int],
    predicted_scores: list[int],
    *,
    classification_threshold: int = AI_THRESHOLD,
) -> dict:
    """Count target-class agreement using a configurable predicted-score cutoff."""
    if len(expected_scores) != len(predicted_scores):
        raise ValueError("expected and predicted test scores must have equal lengths")
    if not 0 <= classification_threshold <= 100:
        raise ValueError(
            "classification threshold must be between zero and one hundred"
        )
    counts = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }
    errors: list[int] = []
    ai_predictions: list[int] = []
    human_predictions: list[int] = []
    for expected, predicted in zip(expected_scores, predicted_scores, strict=True):
        if not 0 <= expected <= 100 or not 0 <= predicted <= 100:
            raise ValueError("test scores must be between zero and one hundred")
        errors.append(abs(predicted - expected))
        expected_ai = expected >= AI_THRESHOLD
        predicted_ai = predicted >= classification_threshold
        if expected_ai and predicted_ai:
            counts["true_positive"] += 1
            ai_predictions.append(predicted)
        elif expected_ai:
            counts["false_negative"] += 1
            ai_predictions.append(predicted)
        elif predicted_ai:
            counts["false_positive"] += 1
            human_predictions.append(predicted)
        else:
            counts["true_negative"] += 1
            human_predictions.append(predicted)
    return {
        **counts,
        "errors": errors,
        "predicted_scores": predicted_scores,
        "ai_predictions": ai_predictions,
        "human_predictions": human_predictions,
    }
