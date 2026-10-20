"""Summarize and persist a live scoring run so its scores can be inspected afterwards.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contracts import is_valid_report, parse_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Written beside the report and the audit probes, all of which version control ignores.
ARTIFACT = PROJECT_ROOT / "build" / "scoring" / "last-run.json"


def _percent(value: Any) -> str:
    """Format a score for the table, or a marker when the score is missing."""
    return f"{value}%" if isinstance(value, int) else "n/a"


def summary_table(reports: dict[str, str], analyses: dict[str, dict[str, Any]]) -> str:
    """Format judge scores and the amount of repository evidence shown for each project."""
    width = max([len(slug) for slug in reports] + [len("repository")])
    lines = [
        f"{'repository':<{width}}  {'judge AI':>8}  {'judge code':>10}  "
        f"{'files':>5}  report"
    ]
    for slug, report in reports.items():
        parsed = parse_report(report)
        analysis = analyses.get(slug) or {}
        coverage = analysis.get("coverage") or {}
        lines.append(
            f"{slug:<{width}}  {_percent(parsed['ai_score']):>8}  "
            f"{_percent(parsed['code_score']):>10}  "
            f"{coverage.get('files_shown', 'n/a')!s:>5}  "
            f"{'ok' if is_valid_report(report) else 'unparseable'}"
        )
    return "\n".join(lines)


def write_artifact(
    path: Path, reports: dict[str, str], analyses: dict[str, dict[str, Any]]
) -> Path:
    """Save full judge reports and prompt-context coverage for later inspection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "written": datetime.now(UTC).isoformat(timespec="seconds"),
                "summary": summary_table(reports, analyses),
                "repositories": {
                    slug: {"report": reports[slug], "analysis": analyses.get(slug) or {}}
                    for slug in reports
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
