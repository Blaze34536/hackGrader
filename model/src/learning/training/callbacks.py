"""Build memory and time-boundary callbacks for one training round."""

import time
from typing import Any

from ..config import CACHE_LIMIT_GB, PATIENCE


def callbacks(dependencies: dict[str, Any], torch, deadline: float | None) -> list[Any]:
    """Build memory reporting, a wall-clock stop, and optional early stopping."""
    trainer_callback = dependencies["TrainerCallback"]
    early_stopping = dependencies["EarlyStoppingCallback"]

    class ReportPeakMemory(trainer_callback):
        """Report CUDA high-water marks and release excess cache during training."""

        def on_log(self, args, state, control, **kwargs) -> None:
            """Print CUDA memory high-water marks with each training log."""
            if torch.cuda.is_available():
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

    result: list[Any] = [ReportPeakMemory()]
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
