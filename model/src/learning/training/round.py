"""Isolate one optimizer round and report known runtime failures to its coordinator."""

import time
from typing import Any

from .. import distributed
from ..pipeline.models import load_continuation_adapter
from .artifacts import RunDeadlineExceeded
from .fit import train_model


def train_round(
    dependencies: dict[str, Any],
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_rows,
    resume_path: str | None,
    adapter_loaded: bool,
    adapter_name: str,
    steps: int,
    round_dir: str,
    deadline: float,
    training_deadline: float,
    learning_rate: float,
):
    """Initialize saved adapter weights, train once, and return round failures for recovery."""
    if resume_path is not None and not adapter_loaded:
        try:
            model = load_continuation_adapter(
                dependencies, model, resume_path, adapter_name
            )
        except SystemExit as exc:
            return None, exc
    try:
        trainer = train_model(
            train_dataset,
            eval_dataset,
            tokenizer,
            model,
            dependencies,
            training_rows=training_rows,
            max_steps=steps,
            round_output_dir=round_dir,
            deadline=deadline,
            learning_rate=learning_rate,
        )
    except RunDeadlineExceeded as exc:
        if (
            "the training deadline" in str(exc)
            and time.monotonic() >= training_deadline
        ):
            return None, exc
        return None, RuntimeError(f"round allocation ended before the cap: {exc}")
    except (
        RuntimeError,
        ValueError,
        OSError,
        TypeError,
        SystemExit,
        KeyboardInterrupt,
    ) as exc:
        if distributed.active():
            raise
        return None, exc
    return trainer, None
