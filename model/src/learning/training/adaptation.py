"""Adjust later optimizer rounds using validation results only."""

from ..config import MIN_ADAPTIVE_LR

FLOOR_EPSILON = 1e-12


def stop_at_learning_rate_floor(
    learning_rate: float,
    previous_metrics: dict | None,
    current_metrics: dict,
) -> bool:
    """Stop strict adaptation after a failed validation plateau at the floor."""
    return (
        learning_rate <= MIN_ADAPTIVE_LR + FLOOR_EPSILON
        and not current_metrics["passed"]
        and previous_metrics is not None
        and _rank(current_metrics) <= _rank(previous_metrics)
    )


def _rank(metrics: dict) -> tuple[int, int, float, float]:
    """Order candidates by gate completion, gate count, accuracy, and error."""
    return (
        int(bool(metrics["passed"])),
        -len(metrics["failed_gates"]),
        float(metrics["balanced_accuracy"]),
        -float(metrics["mean_absolute_error"]),
    )


def next_learning_rate(
    learning_rate: float,
    previous_metrics: dict | None,
    current_metrics: dict,
) -> float:
    """Halve the rate after a validation regression or plateau, with a floor."""
    if learning_rate <= MIN_ADAPTIVE_LR:
        return MIN_ADAPTIVE_LR
    if current_metrics["passed"] or previous_metrics is None:
        return learning_rate
    if _rank(current_metrics) <= _rank(previous_metrics):
        return max(MIN_ADAPTIVE_LR, learning_rate / 2)
    return learning_rate
