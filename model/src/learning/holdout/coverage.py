"""Require a sizeable two-class holdout sample before any GPU work is loaded.

Sampling is deterministic given the rows, so the same check runs again on the scored
sample inside ``evaluate_holdout``. Its value here is failing before model load rather
than after an hour of generation.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from typing import Any

from ..config import MIN_TEST_CLASS_ROWS, MIN_TEST_ROWS
from .intervals import AI_THRESHOLD
from .sampling import holdout_target, stratified, user_prompt


def require_test_coverage(
    rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Require a sizeable two-class final test before loading GPU dependencies."""
    if not rows:
        raise SystemExit(
            "[!] The independent test set is empty; rebuild the dataset before training."
        )
    selected = stratified(rows, limit)
    unique_targets = {user_prompt(row): holdout_target(row) for row in selected}
    human = sum(score < AI_THRESHOLD for score in unique_targets.values())
    ai = sum(score >= AI_THRESHOLD for score in unique_targets.values())
    if not unique_targets:
        raise SystemExit(
            "[!] The independent set has no usable prompts; rebuild the dataset before "
            "training."
        )
    if len(unique_targets) < MIN_TEST_ROWS:
        raise SystemExit(
            f"[!] The independent test set has only {len(unique_targets)} distinct "
            f"selected prompts; at least {MIN_TEST_ROWS} are required. Rebuild the "
            "dataset before training."
        )
    if min(human, ai) < MIN_TEST_CLASS_ROWS:
        raise SystemExit(
            f"[!] The release sample needs at least {MIN_TEST_CLASS_ROWS} examples per "
            f"class below/above {AI_THRESHOLD}%; found {human} human-range and {ai} "
            "AI-range per class."
        )
    return selected
