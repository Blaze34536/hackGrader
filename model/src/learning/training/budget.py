"""Calculate a bounded training-round size from the remaining wall-clock budget."""

from ..config import (
    BUDGET_SAFETY_FACTOR,
    EVAL_STEPS,
    MAX_STEPS_PER_ROUND,
    VALIDATION_SCORING_RESERVE_HOURS,
)


def remaining_training_steps(
    *,
    elapsed_seconds: float,
    seconds_per_step: float,
    budget_hours: float,
    safety_factor: float = BUDGET_SAFETY_FACTOR,
    scoring_reserve_hours: float = VALIDATION_SCORING_RESERVE_HOURS,
    round_limit: int = MAX_STEPS_PER_ROUND,
    eval_steps: int = EVAL_STEPS,
) -> int:
    """Return whole evaluation intervals that fit before reserved scoring time."""
    if round_limit <= 0 or eval_steps <= 0 or round_limit % eval_steps:
        raise ValueError("round_limit must be a positive multiple of eval_steps")
    if (
        elapsed_seconds < 0
        or seconds_per_step <= 0
        or budget_hours <= 0
        or scoring_reserve_hours < 0
        or safety_factor < 1
    ):
        raise ValueError("time budget, step rate, and safety factor must be positive")
    available = budget_hours * 3600 - elapsed_seconds - scoring_reserve_hours * 3600
    if available <= 0:
        return 0
    estimated = int(available // (seconds_per_step * safety_factor))
    bounded = min(round_limit, estimated)
    return bounded // eval_steps * eval_steps
