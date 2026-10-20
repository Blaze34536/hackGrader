"""Configure Trainer placement for quantized models and release evaluation."""

from typing import Any


def eval_configuration_fields(config_type: Any) -> set[str]:
    """Return trainer configuration fields available to held-out checkpoint selection."""
    fields = set(getattr(config_type, "__dataclass_fields__", {}))
    annotations = getattr(getattr(config_type, "__init__", None), "__annotations__", {})
    return fields | set(annotations)


def configure_evaluation(
    config_type: Any, eval_dataset: Any, eval_steps: int
) -> dict[str, Any]:
    """Fail closed unless training can evaluate and restore its best checkpoint."""
    required = {
        "eval_strategy",
        "eval_steps",
        "load_best_model_at_end",
        "metric_for_best_model",
        "save_steps",
    }
    missing = required - eval_configuration_fields(config_type)
    if missing:
        raise SystemExit(
            f"[!] Installed TRL is missing evaluation fields {sorted(missing)}; no training started."
        )
    if eval_steps < 1:
        raise SystemExit(
            "[!] Evaluation and save intervals must be positive; no training started."
        )
    if len(eval_dataset) == 0:
        raise SystemExit(
            "[!] Validation dataset has no usable rows; no training started."
        )
    return {
        "eval_strategy": "steps",
        "eval_steps": eval_steps,
        "save_steps": eval_steps,
        "save_only_model": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }


def ensure_gradient_checkpointing_functions(model) -> bool:
    """Repair Transformers layers flagged for checkpointing but missing the callable."""
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return False
    layers = [
        module
        for module in modules()
        if isinstance(getattr(module, "gradient_checkpointing", None), bool)
        and module.gradient_checkpointing
        and not callable(getattr(module, "_gradient_checkpointing_func", None))
    ]
    if not layers:
        return False
    get_base_model = getattr(model, "get_base_model", None)
    base_model = get_base_model() if callable(get_base_model) else model
    enable = getattr(base_model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise SystemExit(
            "[!] Gradient checkpointing is enabled without its runtime function; "
            "no training started."
        )
    enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    if any(
        isinstance(getattr(layer, "gradient_checkpointing", None), bool)
        and layer.gradient_checkpointing
        and not callable(getattr(layer, "_gradient_checkpointing_func", None))
        for layer in modules()
    ):
        raise SystemExit(
            "[!] Gradient checkpointing remains incomplete after initialization; "
            "no training started."
        )
    print(f"[i] initialized gradient checkpointing for {len(layers)} layer(s).")
    return True


def prepare_training_checkpointing(model) -> None:
    """Attach checkpoint callbacks after Unsloth's train-mode switch toggles layers."""
    enter_training_mode = getattr(model, "for_training", None)
    if callable(enter_training_mode):
        def enter_with_checkpointing(*args, **kwargs):
            result = enter_training_mode(*args, **kwargs)
            ensure_gradient_checkpointing_functions(model)
            return result

        model.for_training = enter_with_checkpointing
        model.for_training()
    else:
        ensure_gradient_checkpointing_functions(model)


def _is_bitsandbytes_quantized(model) -> bool:
    """Detect a bitsandbytes quantized model through an optional PEFT wrapper."""
    candidates = [model]
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        if base_model is not model:
            candidates.append(base_model)
    for candidate in candidates:
        if getattr(candidate, "is_loaded_in_4bit", False) or getattr(
            candidate, "is_loaded_in_8bit", False
        ):
            return True
        method = getattr(candidate, "quantization_method", None)
        if getattr(method, "name", None) == "BITS_AND_BYTES":
            return True
    return False


def configure_quantized_model_placement(model, trainer=None) -> bool:
    """Tell Trainer a PEFT wrapper contains already-placed bitsandbytes weights."""
    if not _is_bitsandbytes_quantized(model):
        return False
    from transformers.utils.quantization_config import QuantizationMethod

    model.quantization_method = QuantizationMethod.BITS_AND_BYTES
    if trainer is not None:
        trainer.place_model_on_device = False
    return True
