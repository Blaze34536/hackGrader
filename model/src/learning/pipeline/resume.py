"""Resolve safe resume/fork checkpoints and their run deadlines."""

import json
import os
import time
from types import SimpleNamespace
from typing import Any

from .. import config, distributed
from ..config import MAX_GATE_CANDIDATES, MAX_STEPS_PER_ROUND
from ..continuation import ATTEMPT_MARKER
from ..resume import validate_resume_checkpoint
from ..resume_fork import verify_fork_outputs
from ..resume_outputs import verify_resume_outputs
from ..resume_runtime import (
    active_input_paths,
    remaining_resume_budget,
    resolve_resume_run,
)
from ..scoring.discovery import checkpoint_directory, checkpoint_run_dir
from ..training.checkpoints import select_round_checkpoint


def prepare_checkpoint_run(
    validate: Any,
    *,
    train_file: str | None,
    validation_file: str | None,
    test_file: str | None,
    dataset_file: str | None,
    resume_checkpoint: str | None,
    fork_checkpoint: str | None,
    run_dir: str,
    release_only: bool = False,
    fresh_repository_test: bool = False,
) -> tuple[Any, Any, Any, str | None, str | None, str]:
    """Validate datasets and checkpoint paths before loading model dependencies."""
    active = active_input_paths(dataset_file, train_file, validation_file, test_file)
    input_options = {}
    if resume_checkpoint is not None:
        input_options["resume_checkpoint"] = resume_checkpoint
    if fork_checkpoint is not None:
        input_options["fork_checkpoint"] = fork_checkpoint
    if release_only:
        input_options["release_only"] = True
    if fresh_repository_test:
        input_options["fresh_repository_test"] = True
    train_rows, validation_rows, test_rows = validate(
        train_file,
        validation_file,
        test_file,
        dataset_file,
        **input_options,
    )
    continuation = resume_checkpoint or fork_checkpoint
    if resume_checkpoint is not None:
        continuation, run_dir = resolve_resume_run(resume_checkpoint, run_dir)
        if os.path.isdir(
            os.path.join(checkpoint_run_dir(continuation), ATTEMPT_MARKER)
        ):
            raise SystemExit(
                "[!] This run already claimed its one-time independent test, so "
                "resuming would spend GPU time only for the test to refuse to run "
                "again. Use `train --continue-run` to finish an interrupted release, "
                "or start a separate run with `train --fork-checkpoint <checkpoint>`."
            )
    elif fork_checkpoint is not None:
        fork_checkpoint = validate_resume_checkpoint(fork_checkpoint)
        if release_only:
            from ..resume_outputs import verify_resume_outputs

            verify_resume_outputs(
                checkpoint_run_dir(fork_checkpoint),
                config.ADAPTER_DIR,
                config.MERGE_DIR,
                active,
            )
        else:
            verify_fork_outputs(
                fork_checkpoint,
                run_dir,
                config.ADAPTER_DIR,
                config.MERGE_DIR,
                active,
            )
        continuation = fork_checkpoint
    return (
        train_rows,
        validation_rows,
        test_rows,
        continuation,
        fork_checkpoint,
        run_dir,
    )


def _initialize_run_directory(
    initialize: Any,
    run_dir: str,
    *,
    inputs: tuple[str, ...],
    forked_from: str | None = None,
    timeout: float,
) -> None:
    """Create the run directory on the main worker and let other workers wait for it."""
    if distributed.is_main():
        initialize(run_dir, forked_from=forked_from, inputs=inputs)
    distributed.await_path(
        os.path.join(run_dir, "run_metadata.json"),
        what="the run metadata",
        timeout=timeout,
    )


def initialize_run(
    initialize: Any,
    *,
    run_dir: str,
    resume_checkpoint: str | None,
    fork_checkpoint: str | None,
    active: tuple[str, ...],
    run_hours: float,
    reserves: float,
    started_at: float,
) -> tuple[float, float | None]:
    """Give new/forked training a fresh cap and existing-run resume its original cap."""
    if fork_checkpoint is not None:
        _initialize_run_directory(
            initialize,
            run_dir,
            inputs=active,
            forked_from=fork_checkpoint,
            timeout=max(600.0, run_hours * 3600),
        )
        return started_at + run_hours * 3600, run_hours * 3600
    if resume_checkpoint is None:
        _initialize_run_directory(
            initialize,
            run_dir,
            inputs=active,
            timeout=max(600.0, run_hours * 3600),
        )
        return started_at + run_hours * 3600, run_hours * 3600
    verify_resume_outputs(run_dir, config.ADAPTER_DIR, config.MERGE_DIR, active)
    remaining = remaining_resume_budget(run_dir, run_hours, reserves)
    return started_at + min(run_hours * 3600, remaining), remaining


def require_selection(
    selection,
    run_dir: str,
    started_at: float,
    block: Any,
    *,
    allow_failed: bool = False,
):
    """Require a scored checkpoint, blocking failed validation unless cap policy allows."""
    if selection is None:
        raise SystemExit(
            "[!] No completed validation-scored checkpoint was produced before the "
            f"runtime cap; no model was merged or registered. Checkpoints remain in {run_dir}."
        )
    if not selection.metrics["passed"]:
        elapsed = (time.monotonic() - started_at) / 3600
        print(
            f"[!] Validation still fails after {elapsed:.1f}h; the best checkpoint is "
            f"retained at {selection.checkpoint_path}. The independent test will not guide training."
        )
        if allow_failed:
            return selection
        block(selection.metrics, checkpoint_path=selection.checkpoint_path)
        raise SystemExit(
            "[!] Validation did not pass; independent-test scoring and deployment stopped."
        )
    return selection


def score_resume_round(
    dependencies, model, tokenizer, validation_rows, checkpoint, deadline
):
    """Validation-score saved candidates from the checkpoint's existing round."""
    checkpoint = validate_resume_checkpoint(checkpoint)
    checkpoint_dir = checkpoint_directory(checkpoint)
    round_dir = os.path.dirname(checkpoint_dir)
    run_dir = checkpoint_run_dir(checkpoint)
    round_number = int(os.path.basename(round_dir).removeprefix("round-"))
    try:
        with open(
            os.path.join(checkpoint_dir, "trainer_state.json"), encoding="utf-8"
        ) as stream:
            best_path = json.load(stream).get("best_model_checkpoint")
    except (OSError, json.JSONDecodeError):
        best_path = None
    return select_round_checkpoint(
        dependencies,
        model,
        tokenizer,
        SimpleNamespace(state=SimpleNamespace(best_model_checkpoint=best_path)),
        validation_rows,
        run_dir=run_dir,
        round_dir=round_dir,
        round_number=round_number,
        candidate_limit=MAX_GATE_CANDIDATES,
        steps_per_round=MAX_STEPS_PER_ROUND,
        deadline=deadline,
    )
