"""Parse target scores and sample unique, class-balanced holdout prompts."""

import random

import contracts as report


def holdout_target(row: dict) -> int:
    """Return the parsed target used for stratified evaluation sampling."""
    return report.parse_report(row["messages"][-1]["content"])["ai_score"] or 0


def user_prompt(row: dict) -> str:
    """Find the user prompt by role, supporting legacy two-message test fixtures."""
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    raise ValueError("training example has no usable user prompt")


def ai_prevalence(rows: list[dict]) -> float | None:
    """Return the machine-authored share of a split's distinct prompts."""
    scores: dict[str, int] = {}
    for row in rows:
        for message in row.get("messages", []):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    scores.setdefault(content, holdout_target(row))
                break
    if not scores:
        return None
    return sum(score >= 40 for score in scores.values()) / len(scores)


def stratified(rows: list[dict], limit: int) -> list[dict]:
    """Select unique prompts from both score classes for validation/test evaluation."""
    if limit <= 0:
        return []
    unique_rows: dict[str, dict] = {}
    prompts_to_targets: dict[str, int] = {}
    for row in rows:
        prompt = user_prompt(row)
        score = holdout_target(row)
        previous = prompts_to_targets.setdefault(prompt, score)
        if previous != score:
            raise ValueError("holdout repeats a prompt with conflicting AI targets")
        unique_rows.setdefault(prompt, row)
    rows = list(unique_rows.values())
    high = [row for row in rows if holdout_target(row) >= 40]
    low = [row for row in rows if holdout_target(row) < 40]
    if not high or not low:
        return (high or low)[:limit]
    rng = random.Random(42)
    rng.shuffle(high)
    rng.shuffle(low)
    high_count = min(len(high), (limit + 1) // 2)
    low_count = min(len(low), limit - high_count)
    remaining = limit - high_count - low_count
    extra_high = min(remaining, len(high) - high_count)
    high_count += extra_high
    low_count += min(remaining - extra_high, len(low) - low_count)
    if high_count == 0 or low_count == 0:
        selected = (high or low)[:limit]
        rng.shuffle(selected)
        return selected
    selected = high[:high_count] + low[:low_count]
    rng.shuffle(selected)
    return selected or rows[:limit]
