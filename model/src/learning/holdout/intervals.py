"""Calculate finite-sample and source-cluster uncertainty intervals."""

from math import sqrt
from typing import Any

AI_THRESHOLD = 40
CONFIDENCE_LEVEL = 0.95
_WILSON_Z = 1.959963984540054
_MIN_INDEPENDENT_CLUSTERS = 10


def _wilson_rate_interval(
    proportion: float, trials: float, z: float = _WILSON_Z
) -> tuple[float, float]:
    """Calculate Wilson bounds for a rate and fractional effective sample size."""
    if not 0.0 <= proportion <= 1.0:
        raise ValueError("proportion must be between zero and one")
    if z <= 0:
        raise ValueError("z must be positive")
    if trials <= 0:
        return 0.0, 1.0
    z_squared = z * z
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    margin = (
        z
        * sqrt(proportion * (1 - proportion) / trials + z_squared / (4 * trials**2))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def wilson_interval(
    successes: int, trials: int, *, z: float = _WILSON_Z
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(
            "binomial successes and trials must satisfy 0 <= successes <= trials"
        )
    if z <= 0:
        raise ValueError("z must be positive")
    if trials == 0:
        return 0.0, 1.0
    return _wilson_rate_interval(successes / trials, trials, z)


def apply_prevalence(
    recall: float, specificity: float, prevalence: float
) -> tuple[float, float]:
    """Return precision and F1 adjusted to a population's positive-class prevalence."""
    if not isinstance(prevalence, (int, float)) or not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must be strictly between zero and one")
    if not 0.0 <= recall <= 1.0 or not 0.0 <= specificity <= 1.0:
        raise ValueError("recall and specificity must be between zero and one")
    false_positive_rate = 1 - specificity
    denominator = prevalence * recall + (1 - prevalence) * false_positive_rate
    adjusted_precision = prevalence * recall / denominator if denominator else 0.0
    adjusted_f1 = (
        2 * adjusted_precision * recall / (adjusted_precision + recall)
        if adjusted_precision + recall
        else 0.0
    )
    return adjusted_precision, adjusted_f1


def _percentile(sorted_values: list[float], probability: float) -> float:
    """Interpolate one percentile from an already sorted sample."""
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def _interval_record(interval: tuple[float, float], method: str) -> dict[str, Any]:
    """Return the shared machine-readable shape for an uncertainty interval."""
    return {"lower": interval[0], "upper": interval[1], "method": method}


def _cluster_adjusted_interval(
    successes: int,
    trials: int,
    cluster_counts: dict[str, tuple[int, int]] | None,
) -> tuple[tuple[float, float], str]:
    """Widen Wilson bounds using observed cluster successes and denominators."""
    row_interval = wilson_interval(successes, trials)
    if cluster_counts is None:
        return row_interval, "Wilson score"
    contributing = [
        (cluster_successes, cluster_trials)
        for cluster_successes, cluster_trials in cluster_counts.values()
        if cluster_trials > 0
    ]
    if any(
        cluster_successes < 0
        or cluster_successes > cluster_trials
        or cluster_trials < 0
        for cluster_successes, cluster_trials in cluster_counts.values()
    ):
        raise ValueError(
            "cluster successes and trials must satisfy 0 <= successes <= trials"
        )
    if (
        sum(cluster_successes for cluster_successes, _ in contributing) != successes
        or sum(cluster_trials for _, cluster_trials in contributing) != trials
    ):
        raise ValueError("cluster counts must sum to the supplied successes and trials")
    if len(contributing) < _MIN_INDEPENDENT_CLUSTERS:
        return (0.0, 1.0), "insufficient source clusters"
    rate = successes / trials if trials else 0.0
    cluster_count = len(contributing)
    residual_sum = sum(
        (cluster_successes - rate * cluster_trials) ** 2
        for cluster_successes, cluster_trials in contributing
    )
    variance = (
        cluster_count / (cluster_count - 1) * residual_sum / (trials * trials)
        if trials
        else 0.0
    )
    margin = _WILSON_Z * sqrt(variance)
    robust = (max(0.0, rate - margin), min(1.0, rate + margin))
    effective_trials = trials**2 / sum(count**2 for _, count in contributing)
    effective = _wilson_rate_interval(rate, effective_trials)
    return (
        min(row_interval[0], robust[0], effective[0]),
        max(row_interval[1], robust[1], effective[1]),
    ), "cluster-adjusted Wilson"
