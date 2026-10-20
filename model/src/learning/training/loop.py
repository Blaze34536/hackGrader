"""Continue isolated training rounds from the best validation-scored adapter."""

import os
import time
from typing import Any

from ..config import (
    ASSUMED_SECONDS_PER_STEP,
    BUDGET_SAFETY_FACTOR,
    FINAL_TEST_RESERVE_HOURS,
    HOURS,
    LORA_LR,
    MAX_GATE_CANDIDATES,
    MAX_STEPS_PER_ROUND,
    RUN_DIR,
    VALIDATION_SCORING_RESERVE_HOURS,
)
from ..holdout import report_release_gates
from ..scoring import VALIDATION_LABEL
from .artifacts import (
    RunDeadlineExceeded,
    measured_seconds_per_step,
    require_time_remaining,
)
from .budget import remaining_training_steps
from .checkpoints import Selection, select_round_checkpoint
from .recovery import handle_interrupted_round
from .round import train_round
from .selection import score_round


def run_training_rounds(
    dependencies: dict[str, Any],
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    started_at: float,
    training_deadline: float,
    validation_deadline: float,
    resume_selection: Selection | None = None,
    starting_checkpoint: str | None = None,
    budget_hours: float = HOURS,
    run_dir: str = RUN_DIR,
    strict_gates: bool = False,
) -> Selection | None:
    """Train fresh optimizer rounds in ``run_dir``, starting from saved weights."""
    torch = dependencies["torch"]
    selection = resume_selection
    resume_path = getattr(resume_selection, "checkpoint_path", starting_checkpoint)
    if resume_selection is not None and resume_selection.metrics["passed"]:
        return resume_selection
    round_number = 0
    if resume_selection is not None:
        report_release_gates(VALIDATION_LABEL, resume_selection.metrics)
    if resume_selection is not None or starting_checkpoint is not None:
        from ..resume import next_round_number

        round_number = next_round_number(run_dir)
    adapter_loaded = resume_path is not None and resume_selection is None
    recovery_attempts = 0
    previous_seconds = ASSUMED_SECONDS_PER_STEP
    cap_reached = False
    learning_rate = LORA_LR

    try:
        while True:
            if strict_gates and selection is not None and selection.metrics["passed"]:
                break
            require_time_remaining(training_deadline, "training")
            elapsed = time.monotonic() - started_at
            measured = measured_seconds_per_step()
            seconds = measured[0] if measured is not None else previous_seconds
            previous_seconds = max(previous_seconds, seconds)
            steps = remaining_training_steps(
                elapsed_seconds=elapsed,
                seconds_per_step=previous_seconds,
                budget_hours=budget_hours,
                scoring_reserve_hours=(
                    VALIDATION_SCORING_RESERVE_HOURS + FINAL_TEST_RESERVE_HOURS
                ),
            )
            if steps <= 0:
                cap_reached = True
                break
            round_number += 1
            round_dir = f"{run_dir}/round-{round_number:02d}"
            if os.path.exists(round_dir):
                print(f"[!] Round output exists: {round_dir}")
                break
            round_deadline = min(
                training_deadline,
                time.monotonic() + steps * previous_seconds * BUDGET_SAFETY_FACTOR,
            )
            print(
                f"[i] round {round_number}; {steps} steps; "
                f"start={'adapter' if resume_path else 'base'}; fresh optimizer."
            )
            trainer, interrupted = train_round(
                dependencies,
                model,
                tokenizer,
                train_dataset,
                eval_dataset,
                train_rows,
                resume_path,
                adapter_loaded,
                f"continuation-{round_number}",
                steps,
                round_dir,
                round_deadline,
                training_deadline,
                learning_rate,
            )
            adapter_loaded = False
            if interrupted is not None:
                previous_selection = selection
                del trainer
                model, tokenizer, selection, learning_rate, at_floor = (
                    handle_interrupted_round(
                        dependencies,
                        model,
                        tokenizer,
                        validation_rows,
                        interrupted,
                        run_dir=run_dir,
                        round_dir=round_dir,
                        round_number=round_number,
                        starting_checkpoint=resume_path,
                        previous_selection=previous_selection,
                        learning_rate=learning_rate,
                        deadline=validation_deadline,
                    )
                )
                if selection is not None:
                    resume_path = selection.checkpoint_path
                    if selection.metrics["passed"]:
                        break
                if strict_gates and at_floor:
                    print(
                        "[!] Adaptive learning-rate floor reached after recovery; "
                        "stopping without spending the independent test."
                    )
                    break
                recovery_attempts += 1
                if recovery_attempts > 1:
                    print(
                        "[!] A second round failure occurred; retaining the best validation checkpoint."
                    )
                    break
                torch.cuda.empty_cache()
                continue
            current, selection, learning_rate, at_floor = score_round(
                dependencies,
                model,
                tokenizer,
                trainer,
                validation_rows,
                selection,
                learning_rate,
                select_round_checkpoint,
                run_dir=run_dir,
                round_dir=round_dir,
                round_number=round_number,
                candidate_limit=MAX_GATE_CANDIDATES,
                steps_per_round=MAX_STEPS_PER_ROUND,
                deadline=validation_deadline,
            )
            print(f"[i] next-round learning rate: {learning_rate:.2e}")
            selection.cap_reached = False
            resume_path = selection.checkpoint_path
            del trainer
            torch.cuda.empty_cache()
            report_release_gates(VALIDATION_LABEL, current.metrics)
            if selection.metrics["passed"]:
                print(
                    f"[i] validation gates passed at checkpoint-{selection.active}; "
                    "stopping continuation before the independent test."
                )
                break
            if strict_gates and at_floor:
                selection.cap_reached = False
                print(
                    "[!] Adaptive learning-rate floor reached with validation "
                    "gates still failing; stopping without spending the independent test."
                )
                break
    except RunDeadlineExceeded as exc:
        cap_reached = "the training deadline" in str(exc)
        print(
            f"[!] Training stopped at its reserved time boundary ({exc}); "
            "the best fully validation-scored checkpoint is retained."
        )
    if selection is not None:
        selection.cap_reached = cap_reached
    return selection
