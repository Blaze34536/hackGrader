"""Require enough distinct, class-balanced prompts in every split.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from data.config import BOUNDARY_BAND, FLAG_THRESHOLD
from data.validation import MAX_TARGET_SHARE

from ..config import (
    MIN_BOUNDARY_CLASS_ROWS,
    MIN_GENERATED_TEST_BOUNDARY_CLASS_ROWS,
    MIN_GENERATED_TEST_CLASS_ROWS,
    MIN_GENERATED_TEST_MIDDLE_SCORE_BAND_ROWS,
    MIN_GENERATED_TEST_ROWS,
    MIN_GENERATED_TEST_SCORE_BAND_ROWS,
    MIN_SCORE_BAND_ROWS,
    MIN_TRAIN_CLASS_ROWS,
    MIN_TRAIN_MIDDLE_SCORE_BAND_ROWS,
    MIN_TRAIN_ROWS,
    MIN_VALIDATION_BOUNDARY_CLASS_ROWS,
    MIN_VALIDATION_CLASS_ROWS,
    MIN_VALIDATION_MIDDLE_SCORE_BAND_ROWS,
    MIN_VALIDATION_ROWS,
    MIN_VALIDATION_SCORE_BAND_ROWS,
)

Requirements = tuple[str, int, int, int, int]


def _unique_targets(
    name: str, rows: list[dict[str, Any]], report_contract: Any
) -> dict[str, int]:
    """Map each distinct prompt to its agreed AI target, rejecting conflicts."""
    unique_targets: dict[str, int] = {}
    for index, row in enumerate(rows):
        try:
            prompt = row["messages"][1]["content"]
            score = report_contract.parse_report(row["messages"][-1]["content"])[
                "ai_score"
            ]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(
                f"{name} row {index} cannot be counted for dataset sufficiency"
            ) from exc
        if (
            not isinstance(prompt, str)
            or score is None
            or not isinstance(score, int)
            or not 0 <= score <= 100
        ):
            raise ValueError(f"{name} row {index} has no usable prompt or AI score")
        previous_score = unique_targets.setdefault(prompt, score)
        if previous_score != score:
            raise ValueError(f"{name} repeats a prompt with conflicting AI targets")
    return unique_targets


def _check_split(
    name: str, unique_targets: dict[str, int], requirements: Requirements
) -> None:
    """Apply one split's floors, reporting the first one it misses."""
    (
        minimum_rows,
        minimum_per_class,
        minimum_per_extreme_band,
        minimum_middle_per_band,
        minimum_boundary_per_side,
    ) = requirements
    if len(unique_targets) < minimum_rows:
        raise ValueError(
            f"{name} has {len(unique_targets)} distinct prompts; at least "
            f"{minimum_rows} are required"
        )

    low_count = sum(score < FLAG_THRESHOLD for score in unique_targets.values())
    middle_count = sum(
        FLAG_THRESHOLD <= score < 60 for score in unique_targets.values()
    )
    high_count = sum(score >= 60 for score in unique_targets.values())
    ai_count = middle_count + high_count
    if min(low_count, ai_count) < minimum_per_class:
        raise ValueError(
            f"{name} has {low_count} human-range and {ai_count} AI-range distinct "
            f"prompts; at least {minimum_per_class} per class are required"
        )
    if min(low_count, high_count) < minimum_per_extreme_band:
        raise ValueError(
            f"{name} has {low_count} low- and {high_count} high-score prompts; "
            f"at least {minimum_per_extreme_band} unique prompts are required in each "
            "extreme score band (<40, 60+)"
        )
    if middle_count < minimum_middle_per_band:
        raise ValueError(
            f"{name} has {middle_count} middle-score prompts; at least "
            f"{minimum_middle_per_band} unique prompts in 40-59 are required"
        )

    boundary_human = sum(
        BOUNDARY_BAND[0] <= score < FLAG_THRESHOLD for score in unique_targets.values()
    )
    boundary_ai = sum(
        FLAG_THRESHOLD <= score <= BOUNDARY_BAND[1] for score in unique_targets.values()
    )
    if min(boundary_human, boundary_ai) < minimum_boundary_per_side:
        raise ValueError(
            f"{name} has {boundary_human} human-side and {boundary_ai} AI-side "
            f"prompts inside the {BOUNDARY_BAND[0]}-{BOUNDARY_BAND[1]} boundary band; "
            f"at least {minimum_boundary_per_side} unique prompts per side are "
            f"required to support the {FLAG_THRESHOLD}% decision boundary"
        )

    target, count = Counter(unique_targets.values()).most_common(1)[0]
    if count / len(unique_targets) > MAX_TARGET_SHARE:
        raise ValueError(
            f"{count / len(unique_targets):.0%} of {name} prompts target {target}%; "
            f"the maximum allowed share is {MAX_TARGET_SHARE:.0%}"
        )


def validate_dataset_sizes(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    report_contract: Any,
) -> None:
    """Require enough distinct prompts and balanced classes in every split."""
    splits = (
        (
            "training",
            train_rows,
            (
                MIN_TRAIN_ROWS,
                MIN_TRAIN_CLASS_ROWS,
                MIN_SCORE_BAND_ROWS,
                MIN_TRAIN_MIDDLE_SCORE_BAND_ROWS,
                MIN_BOUNDARY_CLASS_ROWS,
            ),
        ),
        (
            "validation",
            validation_rows,
            (
                MIN_VALIDATION_ROWS,
                MIN_VALIDATION_CLASS_ROWS,
                MIN_VALIDATION_SCORE_BAND_ROWS,
                MIN_VALIDATION_MIDDLE_SCORE_BAND_ROWS,
                MIN_VALIDATION_BOUNDARY_CLASS_ROWS,
            ),
        ),
        (
            "independent test",
            test_rows,
            (
                MIN_GENERATED_TEST_ROWS,
                MIN_GENERATED_TEST_CLASS_ROWS,
                MIN_GENERATED_TEST_SCORE_BAND_ROWS,
                MIN_GENERATED_TEST_MIDDLE_SCORE_BAND_ROWS,
                MIN_GENERATED_TEST_BOUNDARY_CLASS_ROWS,
            ),
        ),
    )
    for name, rows, requirements in splits:
        _check_split(name, _unique_targets(name, rows, report_contract), requirements)
