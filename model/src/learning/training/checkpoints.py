"""Select checkpoints with bounded, validation-only scoring."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ..config import MAX_GATE_CANDIDATES, MAX_STEPS_PER_ROUND, MIN_TEST_ROWS, SWEEP_ROWS
from ..scoring import VALIDATION_LABEL, checkpoint_steps, evaluate_split, report_sweep
from ..scoring.diagnostics import save_validation_diagnostics
from ..scoring.discovery import adapter_directory, checkpoint_directory
from .artifacts import RunDeadlineExceeded, require_time_remaining


@dataclass
class Selection:
    """A validation-scored adapter checkpoint and its gate metrics."""

    active: int
    trainer_choice: int | None
    moved: bool
    metrics: dict[str, Any]
    checkpoint_path: str
    cap_reached: bool = False
    test_metrics: dict[str, Any] | None = None
    provisional: bool = False
    test_attempted: bool = False


def checkpoint_step(path: str | None) -> int | None:
    """Read the optimizer step from a saved checkpoint directory name."""
    if not path:
        return None
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    checkpoint_path = checkpoint_directory(path)
    name = os.path.basename(os.path.normpath(checkpoint_path))
    if not name.startswith("checkpoint-"):
        return None
    try:
        return int(name.removeprefix("checkpoint-"))
    except ValueError:
        return None


def _candidate_paths(
    available: list[tuple[int, str]], trainer_choice: int | None, limit: int
) -> list[tuple[int, str]]:
    """Bound validation work while always including the trainer's best and latest saves."""
    if not available or limit < 1:
        return []
    if len(available) <= limit:
        return available
    by_local_step = dict(available)
    required: list[int] = []
    if trainer_choice in by_local_step:
        required.append(trainer_choice)
    latest = available[-1][0]
    if latest not in required and limit > 1:
        required.append(latest)
    if limit == 1:
        return [
            (
                required[0] if required else latest,
                by_local_step[required[0] if required else latest],
            )
        ]
    remaining = [step for step, _path in available if step not in required]
    required.extend(
        remaining[:: max(1, len(remaining) // max(1, limit - len(required)))][
            : limit - len(required)
        ]
    )
    chosen = set(required)
    return [(step, path) for step, path in available if step in chosen]


def select_round_checkpoint(
    dependencies: dict[str, Any],
    model,
    tokenizer,
    trainer,
    validation_rows: list[dict[str, Any]],
    *,
    run_dir: str,
    round_dir: str,
    round_number: int,
    candidate_limit: int = MAX_GATE_CANDIDATES,
    steps_per_round: int = MAX_STEPS_PER_ROUND,
    deadline: float | None = None,
) -> Selection:
    """Score at most ``candidate_limit`` checkpoints from this round on validation only."""
    if candidate_limit < 1 or steps_per_round < 1 or round_number < 1:
        raise ValueError(
            "candidate limit, round step stride, and round number must be positive"
        )

    round_dir = os.path.normcase(os.path.realpath(round_dir))
    round_checkpoints = [
        (step, path)
        for step, path in checkpoint_steps(run_dir)
        if os.path.normcase(
            os.path.realpath(os.path.dirname(checkpoint_directory(path)))
        )
        == round_dir
    ]
    local_steps = [(checkpoint_step(path), path) for _step, path in round_checkpoints]
    local_steps = [(step, path) for step, path in local_steps if step is not None]
    if not local_steps:
        raise SystemExit(
            f"[!] Training round {round_number} produced no saved adapter checkpoint; "
            "validation selection stopped."
        )
    local_steps.sort()
    trainer_choice_path = getattr(trainer.state, "best_model_checkpoint", None)
    trainer_adapter = (
        adapter_directory(trainer_choice_path) if trainer_choice_path else None
    )
    trainer_choice = checkpoint_step(trainer_adapter or trainer_choice_path)
    if trainer_choice is not None:
        local_steps = [
            (
                step,
                trainer_adapter if step == trainer_choice and trainer_adapter else path,
            )
            for step, path in local_steps
        ]
    chosen = _candidate_paths(local_steps, trainer_choice, candidate_limit)
    row_limit = SWEEP_ROWS or MIN_TEST_ROWS
    print(
        f"[i] validation-scoring {len(chosen)} of {len(local_steps)} candidate(s) "
        f"from round {round_number} ({row_limit} rows; cap {candidate_limit})."
    )
    original_adapter = getattr(model, "active_adapter", "default")
    if callable(original_adapter):
        original_adapter = original_adapter()
    if isinstance(original_adapter, list):
        original_adapter = original_adapter[0]
    records: list[tuple[int, str, dict[str, Any]]] = []
    torch = dependencies["torch"]
    for local_step, path in chosen:
        logical_step = (round_number - 1) * steps_per_round + local_step
        adapter_name = f"gate-{round_number}-{local_step}"
        try:
            require_time_remaining(deadline, "validation scoring")
            model.load_adapter(path, adapter_name=adapter_name, is_trainable=False)
            model.set_adapter(adapter_name)
            metrics = evaluate_split(
                model,
                tokenizer,
                validation_rows,
                row_limit,
                split=VALIDATION_LABEL,
                deadline=deadline,
            )
            save_validation_diagnostics(run_dir, logical_step, path, metrics)
            metrics.pop("predictions", None)
            records.append((logical_step, VALIDATION_LABEL, metrics))
        except RunDeadlineExceeded:
            if not records:
                raise
            print(f"[!] Validation scoring reached its reserved deadline after {len(records)} of {len(chosen)} candidate(s); selecting the best already scored.", flush=True)
            break
        finally:
            model.set_adapter(original_adapter)
            if adapter_name in getattr(model, "peft_config", {}):
                model.delete_adapter(adapter_name)
            torch.cuda.empty_cache()
    report_sweep(records)
    logical_by_path = {
        path: (round_number - 1) * steps_per_round + local_step
        for local_step, path in chosen
    }
    best_step, _split, best_metrics = max(
        records,
        key=lambda record: (
            int(bool(record[2]["passed"])),
            -len(record[2]["failed_gates"]),
            record[2]["balanced_accuracy"],
            -record[2]["mean_absolute_error"],
            record[0],
        ),
    )
    best_path = next(
        path for path in logical_by_path if logical_by_path[path] == best_step
    )
    trainer_logical_choice = (
        None
        if trainer_choice is None
        else (round_number - 1) * steps_per_round + trainer_choice
    )
    return Selection(
        best_step,
        trainer_logical_choice,
        best_step != trainer_logical_choice,
        best_metrics,
        best_path,
    )
