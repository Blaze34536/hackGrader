"""Load GPU dependencies only after training data has passed preflight checks.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from . import distributed
from .config import BASE_MODEL, GRAD_CHECKPOINTING, LORA_DROPOUT, MAX_LEN
from .inputs import configure_tokenizer


def prepare_environment() -> None:
    """Set environment defaults required before initializing CUDA or Unsloth."""
    if os.name != "nt":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    configured = os.environ.get("HACKGRADER_HF_HOME")
    if configured:
        os.environ.setdefault("HF_HOME", configured)
    elif os.name == "nt" and os.path.isdir("D:\\"):
        os.environ.setdefault("HF_HOME", "D:/huggingface_cache")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"


def model_load_options() -> dict[str, Any]:
    """Place each DDP worker's quantized base model on its own GPU.

    Unsloth defaults to ``device_map="sequential"``, which spreads the 4-bit model
    across every visible GPU. Under DDP each worker must hold a complete replica on
    its own device, or the workers build different parameter sets and DDP aborts.
    """
    if distributed.active():
        return {"device_map": {"": distributed.local_rank()}}
    return {}


def disable_optional_time_limit() -> None:
    """Disable the optional Unsloth timeout wrapper when that hook exists."""
    try:
        import unsloth_zoo.rl_environments
    except ImportError:
        return
    unsloth_zoo.rl_environments.time_limit = lambda _seconds: contextlib.nullcontext()


def load_dependencies() -> dict[str, Any]:
    """Import heavyweight training dependencies after data preflight has succeeded."""
    prepare_environment()
    try:
        import torch
        import unsloth
        from datasets import Dataset
        from transformers import EarlyStoppingCallback, TrainerCallback
        from trl.trainer.sft_config import SFTConfig
        from trl.trainer.sft_trainer import SFTTrainer

        FastLanguageModel = unsloth.FastLanguageModel
    except (ImportError, NotImplementedError, RuntimeError) as exc:
        raise SystemExit(
            "[!] Training runtime dependencies are unavailable or the required Unsloth "
            f"accelerator/runtime could not initialize ({type(exc).__name__}: {exc}). "
            "No training was started."
        ) from None

    disable_optional_time_limit()
    if distributed.active():
        torch.cuda.set_device(distributed.local_rank())
    return {
        "torch": torch,
        "Dataset": Dataset,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "TrainerCallback": TrainerCallback,
        "FastLanguageModel": FastLanguageModel,
    }


def prepare_model(dependencies: dict[str, Any], checkpoint_path: str | None = None):
    """Load the base checkpoint, align its chat template, and attach LoRA adapters."""
    fast_model = dependencies["FastLanguageModel"]
    model, tokenizer = fast_model.from_pretrained(
        model_name=checkpoint_path or BASE_MODEL,
        max_seq_length=MAX_LEN,
        load_in_4bit=True,
        attn_implementation="sdpa",
        **model_load_options(),
    )
    if checkpoint_path:
        if not getattr(model, "peft_config", None):
            raise SystemExit(
                f"[!] Selected checkpoint {checkpoint_path!r} loaded without its "
                "adapter; no scoring, merge, or deployment was performed."
            )
        return model, configure_tokenizer(tokenizer)
    tokenizer = configure_tokenizer(tokenizer)
    model = fast_model.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=GRAD_CHECKPOINTING,
    )
    return model, tokenizer


def warn_if_gpu_busy(torch, min_free_gb: float) -> None:
    """Warn when another process has left insufficient free VRAM for training."""
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    if free < min_free_gb * 1e9:
        print(
            f"[!] Only {free / 1e9:.1f}GB of {total / 1e9:.1f}GB is free before training. "
            "Another process may be using the GPU. Training with too little free memory "
            "can slow down or fail; stop other GPU jobs before starting this run."
        )
