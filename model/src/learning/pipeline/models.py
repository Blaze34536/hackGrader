"""Load validation-selected adapters for another round or final release checks."""

from typing import Any

from ..config import BASE_MODEL, MAX_LEN
from ..runtime import model_load_options, prepare_model


def load_round_model(dependencies: dict[str, Any], checkpoint_path: str | None):
    """Load base weights and, when continuing, initialize a trainable saved adapter."""
    if checkpoint_path is None:
        return prepare_model(dependencies)
    fast_model = dependencies["FastLanguageModel"]
    model, tokenizer = fast_model.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_LEN,
        load_in_4bit=True,
        attn_implementation="sdpa",
        **model_load_options(),
    )
    from ..inputs import configure_tokenizer

    tokenizer = configure_tokenizer(tokenizer)
    try:
        model.load_adapter(
            checkpoint_path,
            adapter_name="continuation",
            is_trainable=True,
        )
        model.set_adapter("continuation")
        fast_model.for_training(model)
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(
            f"[!] Selected adapter {checkpoint_path!r} could not be loaded for "
            f"continuation ({type(exc).__name__}: {exc}); no test scoring or deployment."
        ) from None
    return model, tokenizer


def load_continuation_adapter(
    dependencies: dict[str, Any], model, checkpoint_path: str, adapter_name: str
):
    """Replace round adapters in-place to avoid loading a second quantized base model."""
    existing_adapters = tuple(getattr(model, "peft_config", {}))
    try:
        model.load_adapter(
            checkpoint_path,
            adapter_name=adapter_name,
            is_trainable=True,
        )
        model.set_adapter(adapter_name)
        for existing in existing_adapters:
            if existing != adapter_name:
                model.delete_adapter(existing)
        dependencies["FastLanguageModel"].for_training(model)
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(
            f"[!] Selected adapter {checkpoint_path!r} could not be loaded for "
            f"continuation ({type(exc).__name__}: {exc}); no test scoring or deployment."
        ) from None
    return model


def load_selected_model(dependencies: dict[str, Any], checkpoint_path: str):
    """Load the saved selection as the Unsloth PEFT model the release also merges.

    Unsloth binds its save, merge, and GGUF methods to the model it loads, so the
    adapter has to be loaded through it. Attaching the same adapter to a separately
    loaded base model produces a model whose merge silently writes no weights.
    """
    model, tokenizer = prepare_model(dependencies, checkpoint_path)
    try:
        dependencies["FastLanguageModel"].for_inference(model)
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise SystemExit(
            f"[!] Selected adapter {checkpoint_path!r} could not be prepared for "
            f"final evaluation ({type(exc).__name__}: {exc}); no test scoring or "
            "deployment."
        ) from None
    return model, tokenizer
