"""Authorship and code-quality analysis for bounded repository evidence.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from .analysis import analyze, prompt_context
from .base import (
    ANNOTATION_POINTS,
    BASELINE_AI_SCORE,
    CLEANLINESS_POINTS,
    COMMENT_DENSITY_TARGET,
    COMMENT_POINTS,
    CORROBORATION_CAP,
    CORROBORATION_POINTS,
    DEDUCTION_CAP,
    DOCUMENTATION_EXTS,
    EXPLICIT_ARTIFACT_SCORE,
    EXPLICIT_ARTIFACT_WEIGHT,
    GRADED_AI_CAP,
    MAX_AI_SCORE,
    MAX_QUALITY_SCORE,
    MIN_DOCUMENTED_FUNCTIONS,
    README_IDEAL_CHARS,
    README_MIN_CHARS,
    README_POINTS,
    README_STRUCTURE_IDEAL,
    RESTATEMENT_FLOOR,
    RESTATEMENT_POINTS,
    SATURATION_FLOOR,
    SOURCE_EXTS,
    TRACE_LIMIT,
    TRACES,
    VOLUME_IDEAL_LINES,
)
from .metrics import files_in
from .reports import build_report
from .scoring import score_ai, score_quality
from .signals import AI_SIGNALS, HUMAN_SIGNALS

__all__ = [
    "AI_SIGNALS",
    "ANNOTATION_POINTS",
    "BASELINE_AI_SCORE",
    "CLEANLINESS_POINTS",
    "COMMENT_DENSITY_TARGET",
    "COMMENT_POINTS",
    "CORROBORATION_CAP",
    "CORROBORATION_POINTS",
    "DEDUCTION_CAP",
    "DOCUMENTATION_EXTS",
    "EXPLICIT_ARTIFACT_SCORE",
    "EXPLICIT_ARTIFACT_WEIGHT",
    "GRADED_AI_CAP",
    "HUMAN_SIGNALS",
    "MAX_AI_SCORE",
    "MAX_QUALITY_SCORE",
    "MIN_DOCUMENTED_FUNCTIONS",
    "README_IDEAL_CHARS",
    "README_MIN_CHARS",
    "README_POINTS",
    "README_STRUCTURE_IDEAL",
    "RESTATEMENT_FLOOR",
    "RESTATEMENT_POINTS",
    "SATURATION_FLOOR",
    "SOURCE_EXTS",
    "TRACES",
    "TRACE_LIMIT",
    "VOLUME_IDEAL_LINES",
    "analyze",
    "build_report",
    "files_in",
    "prompt_context",
    "score_ai",
    "score_quality",
]
