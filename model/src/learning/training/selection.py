"""Score one new training round and select from validation evidence only."""

import gc
from typing import Any

from .adaptation import next_learning_rate, stop_at_learning_rate_floor
from .artifacts import RunDeadlineExceeded, require_time_remaining
from .recovery import prefer_selection


def score_round(
    dependencies: dict[str, Any],
    model,
    tokenizer,
    trainer,
    validation_rows: list[dict[str, Any]],
    previous_selection,
    learning_rate: float,
    selector,
    *,
    run_dir: str,
    round_dir: str,
    round_number: int,
    candidate_limit: int,
    steps_per_round: int,
    deadline: float,
):
    """Score a bounded checkpoint set and return the best choice and next rate."""
    gc.collect()
    dependencies["torch"].cuda.empty_cache()
    try:
        require_time_remaining(deadline, "validation scoring")
        current = selector(
            dependencies,
            model,
            tokenizer,
            trainer,
            validation_rows,
            run_dir=run_dir,
            round_dir=round_dir,
            round_number=round_number,
            candidate_limit=candidate_limit,
            steps_per_round=steps_per_round,
            deadline=deadline,
        )
    except RunDeadlineExceeded as exc:
        if previous_selection is None:
            raise
        print(
            f"[!] Validation scoring had no time left for any candidate "
            f"({exc}); retaining the previously selected "
            f"checkpoint-{previous_selection.active}.",
            flush=True,
        )
        return previous_selection, previous_selection, learning_rate, True
    previous_metrics = (
        None if previous_selection is None else previous_selection.metrics
    )
    at_floor = stop_at_learning_rate_floor(
        learning_rate, previous_metrics, current.metrics
    )
    selected = prefer_selection(previous_selection, current)
    rate = next_learning_rate(learning_rate, previous_metrics, current.metrics)
    return current, selected, rate, at_floor
