"""Generate reports for a holdout sample and parse the scores out of them.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import hashlib
from typing import Any

import contracts as report

from ..training.artifacts import require_time_remaining
from .metrics import AI_THRESHOLD
from .sampling import holdout_target, user_prompt

MAX_NEW_TOKENS = 320


def _format_failures(completion: str, parsed: dict[str, Any]) -> list[str]:
    """Classify report-contract failures without retaining generated text."""
    failures = []
    if parsed["ai_score"] is None:
        failures.append("ai_score_missing_or_out_of_range")
    if parsed["code_score"] is None:
        failures.append("code_score_missing_or_out_of_range")
    if len(parsed["summary"]) < 20:
        failures.append("summary_missing_or_too_short")
    if not parsed["valid"]:
        failures.append("report_contract_invalid")
    if "error during chat" in completion.casefold():
        failures.append("generation_error_text")
    return failures


def _prediction_record(
    row: dict[str, Any],
    index: int,
    predicted: int | None,
    classification_threshold: int,
    format_failures: list[str],
    completion: str,
    generated_tokens: int,
) -> dict[str, Any]:
    """Build a prediction record with parsed metadata and generated report text."""
    target = holdout_target(row)
    prompt_digest = hashlib.sha256(user_prompt(row).encode("utf-8")).hexdigest()
    return {
        "row": index,
        "prompt_sha256": prompt_digest,
        "source_type": row.get("source_type", "unknown"),
        "source_kind": row.get("source_kind", "unknown"),
        "source_url": row.get("source_url"),
        "source_groups": row.get("source_groups", []),
        "repository_id": row.get("source_url") or prompt_digest,
        "target_score": target,
        "target_class": "AI-like" if target >= AI_THRESHOLD else "human-like",
        "predicted_score": predicted,
        "predicted_class": (
            None
            if predicted is None
            else "AI-like"
            if predicted >= classification_threshold
            else "human-like"
        ),
        "classification_threshold": classification_threshold,
        "absolute_error": None if predicted is None else abs(predicted - target),
        "parsed": predicted is not None,
        "format_failures": format_failures,
        "generated_output": completion,
        "generated_tokens": generated_tokens,
        "hit_generation_limit": generated_tokens >= MAX_NEW_TOKENS,
    }


def generate_holdout_predictions(
    model,
    tokenizer,
    selected: list[dict[str, Any]],
    *,
    classification_threshold: int = AI_THRESHOLD,
    batch_size: int = 1,
    deadline: float | None = None,
) -> list[dict[str, Any]]:
    """Generate row-level score records, batching GPU work and shrinking batches on OOM."""
    if not 0 <= classification_threshold <= 100:
        raise ValueError(
            "classification threshold must be between zero and one hundred"
        )
    if batch_size < 1:
        raise ValueError("inference batch size must be positive")
    if not selected:
        return []

    import torch
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    predictions: list[dict[str, Any]] = []
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    original_pad_token = getattr(tokenizer, "pad_token", None)
    try:
        if getattr(tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is None:
                raise SystemExit(
                    "[!] Batched inference requires a tokenizer pad or EOS token."
                )
            tokenizer.pad_token = eos_token
        tokenizer.padding_side = "left"

        position = 0
        active_batch_size = min(batch_size, len(selected))
        while position < len(selected):
            require_time_remaining(deadline, "holdout scoring")
            current = selected[position : position + active_batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    row["messages"][:-1], tokenize=False, add_generation_prompt=True
                )
                for row in current
            ]
            inputs = None
            generated = None
            try:
                inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(
                    model.device
                )
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
                    )
            except torch.cuda.OutOfMemoryError:
                if active_batch_size <= 1:
                    raise
                inputs = None
                generated = None
                active_batch_size = max(1, active_batch_size // 2)
                torch.cuda.empty_cache()
                print(
                    f"[!] GPU ran out of memory scoring row {position + 1}; "
                    f"retrying with inference batch size {active_batch_size}.",
                    flush=True,
                )
                continue

            prompt_width = inputs["input_ids"].shape[1]
            for offset, (row, output) in enumerate(
                zip(current, generated, strict=True)
            ):
                completion = tokenizer.decode(
                    output[prompt_width:], skip_special_tokens=True
                )
                parsed = report.parse_report(completion)
                predictions.append(
                    _prediction_record(
                        row,
                        position + offset + 1,
                        parsed["ai_score"],
                        classification_threshold,
                        _format_failures(completion, parsed),
                        completion,
                        output[prompt_width:].shape[-1],
                    )
                )
            position += len(current)
            del inputs, generated
            torch.cuda.empty_cache()
            require_time_remaining(deadline, "holdout scoring")
            if position < len(selected):
                print(
                    f"[i] generated predictions for {position}/{len(selected)} validation rows",
                    flush=True,
                )
    finally:
        tokenizer.padding_side = original_padding_side
        tokenizer.pad_token = original_pad_token
        FastLanguageModel.for_training(model)
    return predictions


def generate_holdout_scores(
    model, tokenizer, selected: list[dict[str, Any]]
) -> tuple[list[int], list[int], int]:
    """Generate aggregate target/prediction arrays for existing gate callers."""
    predictions = generate_holdout_predictions(model, tokenizer, selected)
    parsed = [row for row in predictions if row["parsed"]]
    return (
        [row["target_score"] for row in parsed],
        [row["predicted_score"] for row in parsed],
        len(predictions) - len(parsed),
    )
