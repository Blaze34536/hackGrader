"""Continue isolated training rounds from the best validation-scored adapter."""

import os
import time
from dataclasses import asdict
from typing import Any

from .. import distributed
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
from .artifacts import RunDeadlineExceeded, measured_seconds_per_step
from .budget import remaining_training_steps
from .checkpoints import Selection, select_round_checkpoint
from .recovery import handle_interrupted_round
from .round import train_round
from .selection import score_round


def _decision_path(run_dir: str, name: str) -> str:
    """Return the worker-visible location of one round decision."""
    return os.path.join(run_dir, "workers", name)


def _decision_timeout() -> float:
    """Wait long enough for a full reserved validation sweep plus its release step."""
    return VALIDATION_SCORING_RESERVE_HOURS * 3600 + FINAL_TEST_RESERVE_HOURS * 3600


def _selection_payload(selection: Selection | None) -> dict[str, Any] | None:
    """Describe a selection as JSON so another worker can rebuild it."""
    return None if selection is None else asdict(selection)


def _selection_from(payload: dict[str, Any] | None) -> Selection | None:
    """Rebuild the selection the main worker decided on."""
    return None if payload is None else Selection(**payload)


def _outcome(
    selection: Selection | None,
    learning_rate: float,
    at_floor: bool,
    resume_path: str | None,
    recovery_attempts: int,
    *,
    action: str,
    message: str | None,
    cap_reached: bool = False,
) -> dict[str, Any]:
    """Describe one round's result for every worker."""
    return {
        "selection": _selection_payload(selection),
        "learning_rate": learning_rate,
        "at_floor": at_floor,
        "resume_path": resume_path,
        "recovery_attempts": recovery_attempts,
        "action": action,
        "message": message,
        "cap_reached": cap_reached,
    }


def _next_round_plan(
    *,
    run_dir: str,
    round_number: int,
    started_at: float,
    training_deadline: float,
    previous_seconds: float,
    budget_hours: float,
) -> dict[str, Any]:
    """Size the next round on the main worker and describe it for the other workers."""
    if time.monotonic() >= training_deadline:
        return {
            "stop": True,
            "cap_reached": True,
            "message": (
                "[!] Training stopped at its reserved time boundary; the best fully "
                "validation-scored checkpoint is retained."
            ),
            "steps": 0,
            "seconds": 0.0,
            "observed_seconds": previous_seconds,
            "round_dir": "",
        }
    elapsed = time.monotonic() - started_at
    measured = measured_seconds_per_step()
    observed = measured[0] if measured is not None else previous_seconds
    seconds = max(previous_seconds, observed)
    steps = remaining_training_steps(
        elapsed_seconds=elapsed,
        seconds_per_step=seconds,
        budget_hours=budget_hours,
        scoring_reserve_hours=(
            VALIDATION_SCORING_RESERVE_HOURS + FINAL_TEST_RESERVE_HOURS
        ),
    )
    round_dir = f"{run_dir}/round-{round_number:02d}"
    if steps <= 0:
        return {
            "stop": True,
            "cap_reached": True,
            "message": None,
            "steps": 0,
            "seconds": 0.0,
            "observed_seconds": observed,
            "round_dir": round_dir,
        }
    if os.path.exists(round_dir):
        return {
            "stop": True,
            "cap_reached": False,
            "message": f"[!] Round output exists: {round_dir}",
            "steps": 0,
            "seconds": 0.0,
            "observed_seconds": observed,
            "round_dir": round_dir,
        }
    return {
        "stop": False,
        "cap_reached": False,
        "message": None,
        "steps": steps,
        "seconds": steps * seconds * BUDGET_SAFETY_FACTOR,
        "observed_seconds": observed,
        "round_dir": round_dir,
    }


def _after_round(
    dependencies: dict[str, Any],
    model,
    tokenizer,
    trainer,
    validation_rows: list[dict[str, Any]],
    selection: Selection | None,
    learning_rate: float,
    interrupted: BaseException | None,
    *,
    resume_path: str | None,
    recovery_attempts: int,
    run_dir: str,
    round_dir: str,
    round_number: int,
    validation_deadline: float,
    strict_gates: bool,
) -> tuple[dict[str, Any], Any, Any]:
    """Score this round on validation and describe what every worker does next."""
    if interrupted is not None:
        previous_selection = selection
        (
            model,
            tokenizer,
            selection,
            learning_rate,
            at_floor,
        ) = handle_interrupted_round(
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
        action, message = "continue", None
        if selection is not None:
            resume_path = selection.checkpoint_path
            if selection.metrics["passed"]:
                action = "break"
                message = (
                    f"[i] validation gates passed at checkpoint-{selection.active}; "
                    "stopping continuation before the independent test."
                )
        if action == "continue" and strict_gates and at_floor:
            action = "break"
            message = (
                "[!] Adaptive learning-rate floor reached after recovery; stopping "
                "without spending the independent test."
            )
        if action == "continue":
            recovery_attempts += 1
            if recovery_attempts > 1:
                action = "break"
                message = (
                    "[!] A second round failure occurred; retaining the best "
                    "validation checkpoint."
                )
        return (
            _outcome(
                selection,
                learning_rate,
                at_floor,
                resume_path,
                recovery_attempts,
                action=action,
                message=message,
            ),
            model,
            tokenizer,
        )
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
    selection.cap_reached = False
    resume_path = selection.checkpoint_path
    report_release_gates(VALIDATION_LABEL, current.metrics)
    action = "continue"
    message = f"[i] next-round learning rate: {learning_rate:.2e}"
    if selection.metrics["passed"]:
        action = "break"
        message = (
            f"[i] validation gates passed at checkpoint-{selection.active}; "
            "stopping continuation before the independent test."
        )
    elif strict_gates and at_floor:
        selection.cap_reached = False
        action = "break"
        message = (
            "[!] Adaptive learning-rate floor reached with validation gates still "
            "failing; stopping without spending the independent test."
        )
    return (
        _outcome(
            selection,
            learning_rate,
            at_floor,
            resume_path,
            recovery_attempts,
            action=action,
            message=message,
        ),
        model,
        tokenizer,
    )


def _boundary_outcome(
    selection: Selection | None,
    learning_rate: float,
    at_floor: bool,
    resume_path: str | None,
    recovery_attempts: int,
    error: RunDeadlineExceeded,
) -> dict[str, Any]:
    """Describe a run that stopped because a reserved time boundary was reached."""
    return _outcome(
        selection,
        learning_rate,
        at_floor,
        resume_path,
        recovery_attempts,
        action="break",
        message=(
            f"[!] Training stopped at its reserved time boundary ({error}); "
            "the best fully validation-scored checkpoint is retained."
        ),
        cap_reached="the training deadline" in str(error),
    )


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
    if resume_selection is not None and distributed.is_main():
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
            next_round = round_number + 1
            plan = distributed.shared(
                _decision_path(run_dir, f"plan-{next_round}.json"),
                lambda round_number=next_round, seconds=previous_seconds: (
                    _next_round_plan(
                        run_dir=run_dir,
                        round_number=round_number,
                        started_at=started_at,
                        training_deadline=training_deadline,
                        previous_seconds=seconds,
                        budget_hours=budget_hours,
                    )
                ),
                timeout=_decision_timeout(),
            )
            if plan["stop"]:
                cap_reached = bool(plan["cap_reached"])
                if plan["message"] and distributed.is_main():
                    print(plan["message"])
                break
            round_number = next_round
            steps = int(plan["steps"])
            previous_seconds = max(previous_seconds, float(plan["observed_seconds"]))
            round_dir = str(plan["round_dir"])
            round_deadline = min(
                training_deadline, time.monotonic() + float(plan["seconds"])
            )
            if distributed.is_main():
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
                # Recovery scores saved checkpoints; the trainer's optimizer state is not
                # needed for that and its VRAM is what the scoring pass has to fit in.
                trainer = None
                torch.cuda.empty_cache()
            decision_path = _decision_path(run_dir, f"decision-{round_number}.json")
            if distributed.is_main():
                try:
                    outcome, model, tokenizer = _after_round(
                        dependencies,
                        model,
                        tokenizer,
                        trainer,
                        validation_rows,
                        selection,
                        learning_rate,
                        interrupted,
                        resume_path=resume_path,
                        recovery_attempts=recovery_attempts,
                        run_dir=run_dir,
                        round_dir=round_dir,
                        round_number=round_number,
                        validation_deadline=validation_deadline,
                        strict_gates=strict_gates,
                    )
                except RunDeadlineExceeded as error:
                    outcome = _boundary_outcome(
                        selection,
                        learning_rate,
                        False,
                        resume_path,
                        recovery_attempts,
                        error,
                    )
                    _publish(decision_path, outcome)
                    raise
                except BaseException as error:
                    # Without this the other workers would wait out the whole timeout
                    # for a decision this worker is never going to publish.
                    distributed.publish_failure(decision_path, error)
                    raise
                _publish(decision_path, outcome)
            else:
                outcome = distributed.await_decision(
                    decision_path, timeout=_decision_timeout()
                )
            selection = _selection_from(outcome["selection"])
            learning_rate = float(outcome["learning_rate"])
            resume_path = outcome["resume_path"]
            recovery_attempts = int(outcome["recovery_attempts"])
            if outcome["cap_reached"]:
                cap_reached = True
            if outcome["message"] and distributed.is_main():
                print(outcome["message"])
            del trainer
            torch.cuda.empty_cache()
            if outcome["action"] == "break":
                break
    except RunDeadlineExceeded as error:
        cap_reached = "the training deadline" in str(error)
        if distributed.is_main():
            print(
                f"[!] Training stopped at its reserved time boundary ({error}); "
                "the best fully validation-scored checkpoint is retained."
            )
    if selection is not None:
        selection.cap_reached = cap_reached
    return selection


def _publish(path: str, outcome: dict[str, Any]) -> None:
    """Hand one round outcome to the other workers."""
    if distributed.active():
        distributed.publish(path, outcome)
