"""Score every saved checkpoint of a run by swapping adapters on a resident model.

Training already holds the base weights on the card, so reloading them per checkpoint
would double VRAM use for no benefit. Each swap is undone before the next one, so the
caller's model is left exactly as it was found.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from typing import Any

from .discovery import checkpoint_steps
from .measure import VALIDATION_LABEL, evaluate_split


def sweep_loaded_model(
    dependencies: dict[str, Any],
    model,
    tokenizer,
    run_dir: str,
    rows: list[dict[str, Any]],
    limit: int,
    *,
    split: str = VALIDATION_LABEL,
) -> list[tuple[int, str, dict[str, Any]]]:
    """Score every saved checkpoint of a run against one split."""
    records: list[tuple[int, str, dict[str, Any]]] = []
    original_adapter = getattr(model, "active_adapter", "default")
    if callable(original_adapter):
        original_adapter = original_adapter()
    if isinstance(original_adapter, list):
        original_adapter = original_adapter[0]
    for step, path in checkpoint_steps(run_dir):
        adapter_name = f"score-{step}"
        try:
            model.load_adapter(path, adapter_name=adapter_name, is_trainable=False)
            model.set_adapter(adapter_name)
            records.append(
                (
                    step,
                    split,
                    evaluate_split(model, tokenizer, rows, limit, split=split),
                )
            )
        finally:
            model.set_adapter(original_adapter)
            if adapter_name in getattr(model, "peft_config", {}):
                model.delete_adapter(adapter_name)
            dependencies["torch"].cuda.empty_cache()
    return records
