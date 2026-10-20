"""Score bands and ownership classes used to partition candidate data.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from ..config import BOUNDARY_BAND, FLAG_THRESHOLD

_SCORE_CLASSES = ("human", "ai")
_SCORE_BANDS = ("0-39", "40-59", "60-100")
_PROXIMITY_CLASSES = ("human_boundary", "ai_boundary")
_CLASS_LABELS = (*_SCORE_CLASSES, *_PROXIMITY_CLASSES)
_CLASS_DESCRIPTIONS = {
    "human": "human-range",
    "ai": "AI-range",
    "human_boundary": "human-side boundary",
    "ai_boundary": "AI-side boundary",
}


def class_labels(score: int) -> tuple[str, ...]:
    """Return every ownership class one target score satisfies."""
    within_band = BOUNDARY_BAND[0] <= score <= BOUNDARY_BAND[1]
    if score < FLAG_THRESHOLD:
        return ("human", "human_boundary") if within_band else ("human",)
    return ("ai", "ai_boundary") if within_band else ("ai",)


def score_band(score: int) -> str:
    """Classify one parsed AI score using the training contract's three bands."""
    if score < FLAG_THRESHOLD:
        return "0-39"
    if score < 60:
        return "40-59"
    return "60-100"
