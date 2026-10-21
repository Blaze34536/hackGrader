"""Configure and run bounded supervised fine-tuning rounds."""

import inspect
import time
from typing import Any

from .. import distributed
from ..config import (
    ACCUM,
    EPOCHS,
    EVAL_STEPS,
    LORA_LR,
    MAX_LEN,
    MICRO_BATCH,
    OPTIM,
    RUN_DIR,
    WEIGHT_DECAY,
)
from ..inputs import build_collator
from .artifacts import configure_evaluation, require_time_remaining
from .callbacks import callbacks as _callbacks
from .trainer_config import (
    configure_quantized_model_placement,
    prepare_training_checkpointing,
)


def _configuration(
    config_type,
    eval_dataset,
    torch,
    *,
    round_output_dir: str | None = None,
    learning_rate: float = LORA_LR,
):
    """Build round-local settings with periodic validation and best-checkpoint restore."""
    evaluation = configure_evaluation(config_type, eval_dataset, EVAL_STEPS)
    fields = set(getattr(config_type, "__dataclass_fields__", {}))
    packing = {"packing": False} if "packing" in fields else {}
    init_parameters = getattr(
        getattr(config_type, "__init__", None), "__annotations__", {}
    )
    sft_fields = {
        "dataset_text_field": "text",
        "max_seq_length": MAX_LEN,
    }
    if fields and not {"dataset_text_field", "max_seq_length"} <= fields:
        if isinstance(config_type, type) and hasattr(config_type, "model_fields"):
            config_fields = config_type.model_fields
        elif isinstance(config_type, type) and hasattr(config_type, "__init__"):
            config_fields = getattr(config_type.__init__, "__annotations__", {})
        else:
            config_fields = {}
        sft_fields = {
            name: value
            for name, value in sft_fields.items()
            if name in fields or name in config_fields or name in init_parameters
        }
    if not fields and init_parameters:
        try:
            parameters = inspect.signature(config_type).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        has_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
        sft_fields = {
            name: value
            for name, value in sft_fields.items()
            if name in init_parameters or has_kwargs
        }
    if not sft_fields:
        raise SystemExit(
            "[!] Installed TRL lacks the supported SFT configuration interface; "
            "no training started."
        )

    bf16 = torch.cuda.is_bf16_supported()
    accumulation = ACCUM
    if distributed.active():
        # Keep the optimizer batch equal to the single-GPU batch: extra workers buy
        # speed, not a silently different learning problem.
        accumulation = max(1, ACCUM // distributed.world_size())
        if distributed.is_main():
            print(
                f"[i] {distributed.world_size()} workers; gradient accumulation "
                f"{accumulation} (single-GPU value {ACCUM})."
            )
    return config_type(
        **sft_fields,
        do_train=True,
        per_device_train_batch_size=MICRO_BATCH,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation,
        num_train_epochs=EPOCHS,
        warmup_ratio=0.05,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        optim=OPTIM,
        fp16=not bf16,
        bf16=bf16,
        report_to="none",
        logging_steps=10,
        seed=42,
        output_dir=round_output_dir or RUN_DIR,
        overwrite_output_dir=False,
        ignore_data_skip=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        **evaluation,
        **packing,
    )


def train_model(
    train_dataset,
    eval_dataset,
    tokenizer,
    model,
    dependencies: dict[str, Any],
    *,
    training_rows: list[dict[str, Any]],
    max_steps: int,
    round_output_dir: str,
    deadline: float,
    learning_rate: float = LORA_LR,
) -> Any:
    """Train one fresh-optimizer round initialized by the already-loaded adapter weights."""
    if max_steps < EVAL_STEPS or max_steps % EVAL_STEPS:
        raise ValueError("max_steps must be a positive multiple of EVAL_STEPS")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not round_output_dir:
        raise ValueError("round_output_dir is required to isolate round checkpoints")
    distributed.prepare_round_directory(round_output_dir, timeout=3600.0)

    torch = dependencies["torch"]
    config = _configuration(
        dependencies["SFTConfig"],
        eval_dataset,
        torch,
        round_output_dir=round_output_dir,
        learning_rate=learning_rate,
    )
    config.max_steps = max_steps
    config.max_time = max(1.0, deadline - time.monotonic())

    quantized = configure_quantized_model_placement(model)
    trainer = dependencies["SFTTrainer"](
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=build_collator(tokenizer, training_rows),
        callbacks=_callbacks(dependencies, torch, deadline),
        args=config,
    )
    prepare_training_checkpointing(trainer.model)
    if quantized:
        trainer.place_model_on_device = False
    torch.cuda.empty_cache()
    require_time_remaining(deadline, "training round")
    trainer.train()

    from .artifacts import report_training_rate

    if distributed.is_main():
        report_training_rate(trainer)
    if trainer.state.global_step <= 0:
        raise SystemExit(
            "[!] The trainer completed no optimizer steps; deployment stopped."
        )
    if not trainer.state.best_model_checkpoint:
        raise SystemExit(
            "[!] No best validation checkpoint was selected; deployment stopped."
        )
    return trainer
