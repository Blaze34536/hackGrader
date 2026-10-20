"""Holdout evaluation and the release gates that stand between training and deployment.

    from .holdout import evaluate_holdout, require_test_coverage, summarize_holdout

Implementation modules:

``metrics``      measures error, separation, and per-class reliability
``gates``        checks all release floors and lists every failure
``coverage``     checks the holdout before loading the model
``generation``   scores a sample and reads its generated reports
``enforcement``  stops callers that require a passing result

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from .coverage import require_test_coverage
from .enforcement import check_holdout, score_holdout
from .evaluation import evaluate_holdout
from .gates import (
    DEFAULT_SPLIT,
    UNPARSEABLE_RATE_LIMIT,
    assess_uncertainty,
    format_gate_failures,
    report_release_gates,
    summarize_holdout,
)
from .generation import generate_holdout_predictions, generate_holdout_scores
from .intervals import AI_THRESHOLD, CONFIDENCE_LEVEL, apply_prevalence, wilson_interval
from .lineage import lineage_clusters
from .metric_calculations import confusion
from .metrics import sample_metrics
from .sampling import ai_prevalence, holdout_target, stratified, user_prompt

__all__ = [
    "AI_THRESHOLD",
    "CONFIDENCE_LEVEL",
    "DEFAULT_SPLIT",
    "UNPARSEABLE_RATE_LIMIT",
    "ai_prevalence",
    "apply_prevalence",
    "assess_uncertainty",
    "check_holdout",
    "confusion",
    "evaluate_holdout",
    "format_gate_failures",
    "generate_holdout_predictions",
    "generate_holdout_scores",
    "holdout_target",
    "lineage_clusters",
    "report_release_gates",
    "require_test_coverage",
    "sample_metrics",
    "score_holdout",
    "stratified",
    "summarize_holdout",
    "user_prompt",
    "wilson_interval",
]
