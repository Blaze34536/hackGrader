"""Convert weighted detector findings into bounded percentage scores.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from .base import (
    BASELINE_AI_SCORE,
    DEDUCTION_CAP,
    EXPLICIT_ARTIFACT_SCORE,
    EXPLICIT_ARTIFACT_WEIGHT,
    GRADED_AI_CAP,
    MAX_AI_SCORE,
    MAX_QUALITY_SCORE,
)


def score_ai(findings: list[dict], human: list[dict]) -> int:
    """Combine machine-like and human-trace findings into a 0-100 score."""
    graded = sum(finding["weight"] for finding in findings)
    score = BASELINE_AI_SCORE + min(graded, GRADED_AI_CAP)
    score -= sum(finding["weight"] for finding in human)
    if any(finding["weight"] >= EXPLICIT_ARTIFACT_WEIGHT for finding in findings):
        score = max(score, EXPLICIT_ARTIFACT_SCORE)
    return max(0, min(MAX_AI_SCORE, score))


def score_quality(findings: list[dict]) -> int:
    """Subtract bounded quality deductions from the maximum score."""
    total = min(sum(finding["points"] for finding in findings), DEDUCTION_CAP)
    return max(0, min(MAX_QUALITY_SCORE, MAX_QUALITY_SCORE - total))
