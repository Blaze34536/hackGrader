"""Prepare model-visible conversations and prove assistant-only training loss."""

from __future__ import annotations

from typing import Any

import contracts as report

from .config import ASSISTANT_ONLY_LOSS, MAX_LEN, RESPONSE_TEMPLATE
from .holdout.sampling import ai_prevalence, holdout_target, stratified, user_prompt

__all__ = [
    "ai_prevalence",
    "build_collator",
    "build_dataset",
    "configure_tokenizer",
    "holdout_target",
    "label_spread",
    "prove_masking",
    "render",
    "stratified",
    "user_prompt",
]


def configure_tokenizer(tokenizer):
    """Apply the Qwen conversation template used by the serving contract."""
    from unsloth.chat_templates import get_chat_template

    return get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
        mapping={
            "role": "role",
            "content": "content",
            "user": "user",
            "assistant": "assistant",
            "system": "system",
            "tool_calls": "tool_calls",
            "name": "name",
        },
    )


def render(tokenizer, messages: list[dict]) -> str:
    """Render the full conversation with the inference-compatible chat template."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def build_dataset(
    tokenizer,
    rows: list[dict],
    *,
    shuffle: bool = False,
    what: str = "train",
) -> Any:
    """Render chats, prove assistant-only boundaries, and reject any truncation."""
    kept: list[str] = []
    dropped = {"snippet": 0, "repository": 0}
    longest = 0
    for row in rows:
        text = render(tokenizer, row["messages"])
        _verify_rendered_target(text, row)
        length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        longest = max(longest, length)
        if length <= MAX_LEN:
            kept.append(text)
        else:
            dropped[str(row["source_kind"])] += 1
    _verify_dataset_size(kept, dropped)
    print(f"[i] {what}: kept {len(kept)} row(s); longest row {longest} token(s)")
    from datasets import Dataset

    dataset = Dataset.from_list([{"text": text} for text in kept])
    return dataset.shuffle(seed=42) if shuffle else dataset


def _verify_rendered_target(text: str, row: dict) -> None:
    """Require one assistant header and exact target text after rendering."""
    if text.count(RESPONSE_TEMPLATE) != 1:
        raise SystemExit(
            "[!] A row does not render exactly one assistant header, so completion-only "
            "loss cannot separate the report from repository text. Rebuild the dataset."
        )
    target = row["messages"][-1]["content"][:64]
    if not text.split(RESPONSE_TEMPLATE, 1)[1].startswith(target):
        raise SystemExit(
            "[!] The rendered assistant span differs from its target, so the masked "
            "loss would teach the wrong sequence. Rebuild the dataset."
        )


def _verify_dataset_size(kept: list[str], dropped: dict[str, int]) -> None:
    """Reject empty or truncated sources instead of fitting incomplete examples."""
    if not kept:
        raise SystemExit(f"No rows fit MAX_LEN={MAX_LEN}; raise HACKGRADER_MAX_LEN.")
    if dropped["repository"] or dropped["snippet"]:
        raise SystemExit(
            f"[!] {dropped['repository']} repository and {dropped['snippet']} snippet row(s) "
            f"exceed MAX_LEN={MAX_LEN}. Refusing to train on an incomplete dataset."
        )


def build_collator(tokenizer, training_rows: list[dict[str, Any]]):
    """Construct the completion-only collator and prove it masks repository prompt tokens."""
    if not ASSISTANT_ONLY_LOSS:
        raise SystemExit(
            "[!] Assistant-only loss is required: repository text must not be trained as output."
        )
    try:
        from trl import DataCollatorForCompletionOnlyLM

        collator = DataCollatorForCompletionOnlyLM(
            response_template=RESPONSE_TEMPLATE, tokenizer=tokenizer
        )
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"[!] The assistant-only collator is unavailable ({type(exc).__name__}: {exc}). "
            "Training without it would model the repository instead of the report."
        ) from None
    prove_masking(tokenizer, collator, training_rows)
    return collator


def prove_masking(tokenizer, collator, training_rows: list[dict[str, Any]]) -> None:
    """Check that trainable labels start immediately after a single assistant header."""
    rendered = [render(tokenizer, row["messages"]) for row in training_rows[:2]]
    batch = collator([tokenizer(text) for text in rendered])
    header = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    for index, labels in enumerate(batch["labels"]):
        trained = [position for position, label in enumerate(labels) if label != -100]
        if not trained:
            raise SystemExit(
                "[!] The collator masked the whole row; nothing would be learned."
            )
        first = trained[0]
        before = batch["input_ids"][index][max(0, first - len(header)) : first]
        if tokenizer.decode(before) != RESPONSE_TEMPLATE:
            raise SystemExit(
                "[!] The first loss token is not after the assistant header."
            )


def label_spread(rows: list[dict]) -> str:
    """Summarize target scores by band before the expensive model load."""
    buckets = {"0-19": 0, "20-39": 0, "40-59": 0, "60-79": 0, "80-100": 0}
    for row in rows:
        score = report.parse_report(row["messages"][-1]["content"])["ai_score"]
        if score is None:
            continue
        for name, lower in (
            ("80-100", 80),
            ("60-79", 60),
            ("40-59", 40),
            ("20-39", 20),
        ):
            if score >= lower:
                buckets[name] += 1
                break
        else:
            buckets["0-19"] += 1
    return " | ".join(f"{name}%: {count}" for name, count in buckets.items())
