"""Turn holdout metrics into a stop, so a rejected model is never deployed.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from typing import Any

from .coverage import require_test_coverage
from .evaluation import evaluate_holdout
from .gates import DEFAULT_SPLIT, format_gate_failures, summarize_holdout
from .generation import generate_holdout_scores


def score_holdout(
    expected_scores: list[int],
    predicted_scores: list[int],
    *,
    unparsed: int = 0,
    split: str = DEFAULT_SPLIT,
    prevalence: float | None = None,
) -> dict[str, Any]:
    """Print every failed gate, then fail closed with a nonzero exit."""
    metrics = evaluate_holdout(
        expected_scores,
        predicted_scores,
        unparsed=unparsed,
        split=split,
        prevalence=prevalence,
    )
    print(summarize_holdout(metrics))
    if metrics["violations"]:
        for failure in format_gate_failures(metrics):
            print(f"[!] {failure}")
        first_failure = metrics["violations"][0]["message"]
        raise SystemExit(first_failure)
    return metrics


def check_holdout(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    limit: int,
    *,
    split: str = DEFAULT_SPLIT,
    prevalence: float | None = None,
) -> dict[str, Any]:
    """Run generated reports on a source-disjoint holdout and reject weak models."""
    selected = require_test_coverage(rows, limit)
    expected_scores, predicted_scores, unparsed = generate_holdout_scores(
        model, tokenizer, selected
    )
    return score_holdout(
        expected_scores,
        predicted_scores,
        unparsed=unparsed,
        split=split,
        prevalence=prevalence,
    )
