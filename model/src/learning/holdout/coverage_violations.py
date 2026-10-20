"""Build holdout coverage and class-support gate diagnostics."""

from typing import Any

from ..config import MIN_TEST_CLASS_ROWS, MIN_TEST_ROWS
from .gates import UNPARSEABLE_RATE_LIMIT


def coverage_violations(total: int, unparsed: int, label: str) -> list[dict[str, str]]:
    """Report too few evaluated examples and too many unparseable reports."""
    violations = []
    if total < MIN_TEST_ROWS:
        violations.append(
            {
                "gate": "coverage",
                "message": f"[!] Only {total} {label} examples were evaluated; at least {MIN_TEST_ROWS} are required. Not deploying.",
            }
        )
    rate = unparsed / total if total else 1.0
    if rate > UNPARSEABLE_RATE_LIMIT:
        violations.append(
            {
                "gate": "unparsed",
                "message": f"[!] {rate:.0%} of {label} rows produced unparseable scores; the maximum is {UNPARSEABLE_RATE_LIMIT:.0%}. Not deploying.",
            }
        )
    return violations


def class_support_violations(
    metrics: dict[str, Any], label: str
) -> list[dict[str, str]]:
    """Report a parsed sample that cannot support one of its two classes."""
    if min(metrics["ai_support"], metrics["human_support"]) >= MIN_TEST_CLASS_ROWS:
        return []
    return [
        {
            "gate": "class_support",
            "message": f"[!] Parsed {label} needs at least {MIN_TEST_CLASS_ROWS} examples per class; found {metrics['human_support']} human-range and {metrics['ai_support']} AI-range. Not deploying.",
        }
    ]
