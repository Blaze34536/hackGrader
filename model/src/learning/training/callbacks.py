"""Build memory and time-boundary callbacks for one training round."""

import time
from typing import Any

from .. import distributed
from ..config import CACHE_LIMIT_GB, PATIENCE


def callbacks(dependencies: dict[str, Any], torch, deadline: float | None) -> list[Any]:
    """Build memory reporting, a wall-clock stop, and optional early stopping."""
    trainer_callback = dependencies["TrainerCallback"]
    early_stopping = dependencies["EarlyStoppingCallback"]

    class ReportPeakMemory(trainer_callback):
        """Report CUDA high-water marks and release excess cache during training."""

        def on_log(self, args, state, control, **kwargs) -> None:
            """Print CUDA memory high-water marks with each training log."""
            if torch.cuda.is_available() and distributed.is_main():
                print(
                    f"[i] VRAM: allocated {torch.cuda.max_memory_allocated() / 1e9:.1f}GB; reserved {torch.cuda.max_memory_reserved() / 1e9:.1f}GB."
                )

        def on_step_end(self, args, state, control, **kwargs) -> None:
            """Release unused CUDA cache when it exceeds the configured limit."""
            if (
                torch.cuda.is_available()
                and torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
                > CACHE_LIMIT_GB * 1e9
            ):
                torch.cuda.empty_cache()

    class ReportTrainValGap(trainer_callback):
        """Print the train-to-validation gap beside each evaluation.

        A train loss that keeps falling while the validation loss stops is the signature
        of fitting the rows rather than the reading. The two numbers are already recorded
        in ``trainer_state.json``; printing the gap makes it readable while the run is
        still going, instead of after a deployment that trusted the training curve.
        """

        def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
            """Report train loss, validation loss, and their difference."""
            eval_loss = (metrics or {}).get("eval_loss")
            if eval_loss is None or not distributed.is_main():
                return
            history = getattr(state, "log_history", None) or []
            train_losses = [entry["loss"] for entry in history if "loss" in entry]
            if not train_losses:
                return
            train_loss = train_losses[-1]
            print(
                f"[i] step {state.global_step}: train loss {train_loss:.4f}; "
                f"eval loss {eval_loss:.4f}; gap {eval_loss - train_loss:+.4f}."
            )

    result: list[Any] = [ReportPeakMemory(), ReportTrainValGap()]
    if deadline is not None:

        class StopAtDeadline(trainer_callback):
            """Stop gracefully at an optimizer-step boundary when time expires."""

            def on_step_end(self, args, state, control, **kwargs) -> None:
                """Request a stop before another optimizer step starts."""
                if time.monotonic() >= deadline:
                    control.should_training_stop = True

        result.append(StopAtDeadline())
    if PATIENCE > 0:
        result.append(early_stopping(early_stopping_patience=PATIENCE))
    return result
