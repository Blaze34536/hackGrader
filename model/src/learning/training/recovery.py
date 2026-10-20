"""Recover saved validation checkpoints after an optimizer round fails."""

from types import SimpleNamespace
from typing import Any

from ..config import MAX_GATE_CANDIDATES, MAX_STEPS_PER_ROUND
from ..pipeline.models import load_continuation_adapter
from ..resume import score_resume_checkpoint
from .adaptation import next_learning_rate, stop_at_learning_rate_floor
from .checkpoints import Selection, select_round_checkpoint


def prefer_selection(previous: Selection | None, current: Selection) -> Selection:
    """Retain the strongest validation result without consulting independent-test data."""
    if previous is None or current.metrics["passed"]:
        return current
    if previous.metrics["passed"]:
        return previous
    current_key = (
        -len(current.metrics["failed_gates"]),
        current.metrics["balanced_accuracy"],
        -current.metrics["mean_absolute_error"],
        current.active,
    )
    previous_key = (
        -len(previous.metrics["failed_gates"]),
        previous.metrics["balanced_accuracy"],
        -previous.metrics["mean_absolute_error"],
        previous.active,
    )
    return current if current_key > previous_key else previous


def handle_interrupted_round(
    dependencies: dict[str, Any],
    model,
    tokenizer,
    validation_rows: list[dict[str, Any]],
    interrupted: BaseException,
    *,
    run_dir: str,
    round_dir: str,
    round_number: int,
    starting_checkpoint: str | None,
    previous_selection: Selection | None,
    learning_rate: float,
    deadline: float,
) -> tuple[Any, Any, Selection, float, bool]:
    """Recover saved weights and adapt using validation-only evidence."""
    print(
        f"[!] Training round {round_number} stopped "
        f"({type(interrupted).__name__}: {interrupted}); checking saved weights."
    )
    del interrupted
    dependencies["torch"].cuda.empty_cache()
    model, tokenizer, recovered = recover_interrupted_round(
        dependencies,
        validation_rows,
        run_dir=run_dir,
        round_dir=round_dir,
        round_number=round_number,
        model=model,
        tokenizer=tokenizer,
        starting_checkpoint=starting_checkpoint,
        previous_selection=previous_selection,
        deadline=deadline,
    )
    if recovered is None:
        print(
            "[!] Training failed before any checkpoint could be validation-scored and "
            "no earlier selection exists; the run coordinator decides whether to retry.",
            flush=True,
        )
        return model, tokenizer, previous_selection, learning_rate, True
    selection = prefer_selection(previous_selection, recovered)
    previous_metrics = (
        None if previous_selection is None else previous_selection.metrics
    )
    stop = stop_at_learning_rate_floor(
        learning_rate, previous_metrics, recovered.metrics
    )
    if selection.metrics["passed"]:
        stop = True
    rate = next_learning_rate(learning_rate, previous_metrics, recovered.metrics)
    selection.cap_reached = False
    return model, tokenizer, selection, rate, stop


def recover_interrupted_round(
    dependencies: dict[str, Any],
    validation_rows: list[dict[str, Any]],
    *,
    run_dir: str,
    round_dir: str,
    round_number: int,
    model,
    tokenizer,
    starting_checkpoint: str | None,
    previous_selection: Selection | None,
    deadline: float,
) -> tuple[Any, Any, Selection | None]:
    """Reuse the placed base model and score available checkpoints on validation."""
    if starting_checkpoint is not None:
        adapter_name = f"recovery-{round_number}"
        model = load_continuation_adapter(
            dependencies, model, starting_checkpoint, adapter_name
        )
    trainer = SimpleNamespace(state=SimpleNamespace(best_model_checkpoint=None))
    try:
        selection = select_round_checkpoint(
            dependencies,
            model,
            tokenizer,
            trainer,
            validation_rows,
            run_dir=run_dir,
            round_dir=round_dir,
            round_number=round_number,
            candidate_limit=MAX_GATE_CANDIDATES,
            steps_per_round=MAX_STEPS_PER_ROUND,
            deadline=deadline,
        )
    except (SystemExit, RuntimeError, ValueError, OSError) as exc:
        selection = None
        print(
            f"[!] Validation sweep of round {round_number} failed "
            f"({type(exc).__name__}: {exc}); no checkpoint was scored.",
            flush=True,
        )
        del exc
    if (
        selection is None
        and previous_selection is None
        and starting_checkpoint is not None
    ):
        try:
            selection = score_resume_checkpoint(
                dependencies,
                model,
                tokenizer,
                validation_rows,
                starting_checkpoint,
                100,
                deadline,
            )
        except (SystemExit, RuntimeError, ValueError, OSError) as exc:
            selection = None
            print(
                f"[!] Resumable checkpoint scoring failed "
                f"({type(exc).__name__}: {exc}); no checkpoint was scored.",
                flush=True,
            )
            del exc
    if selection is None and previous_selection is not None:
        selection = previous_selection
    if selection is not None:
        selection.cap_reached = False
    if selection is None:
        print(
            "[!] Nothing could be validation-scored for this round and no earlier "
            f"selection exists; checkpoints remain in {run_dir}.",
            flush=True,
        )
        return model, tokenizer, previous_selection
    return model, tokenizer, prefer_selection(previous_selection, selection)
