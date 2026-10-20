"""Prepare annotated score records for rendering in the PDF report.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import json
import re
import time
from pathlib import Path
from typing import Any

from grading.coverage import describe_coverage

from .layout import percent, title_lines, wrap_by_width

FAILURE_LINES = 2
EVIDENCE_LIMIT = 6
DEDUCTION_LIMIT = 6
_IMPACT_PATTERN = re.compile(r"^-(\d+)%")
MODEL_RECORD = (
    Path(__file__).resolve().parents[2] / "model" / "data" / "deployed_model.json"
)


def grader_model_tag(record: Path = MODEL_RECORD) -> str:
    """Read the locally recorded grader build tag, or nothing when it is unavailable.

    The release record is read from disk only: this package does not call Ollama or
    GitHub, and a missing or unreadable record simply leaves the line off the cover.
    """
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    tag = data.get("tag") if isinstance(data, dict) else None
    return tag if isinstance(tag, str) else ""


def coverage_cell(coverage: dict[str, Any]) -> str:
    """Summarize reproduced and discovered file counts for the index table."""
    shown = coverage.get("files_shown")
    if shown is None:
        return ""
    found = coverage.get("files_found", shown)
    cell = f"{shown} of {found} files" if found != shown else f"{shown} files"
    return f"{cell}, partial" if coverage.get("truncated") else cell


def _impact(line: Any) -> int:
    """Return the percentage a deduction cites, or -1 when it has no impact."""
    match = _IMPACT_PATTERN.match(str(line).strip())
    return int(match.group(1)) if match else -1


def capped(items: list[str], limit: int) -> tuple[list[str], int]:
    """Trim a panel's items to its limit and report how many were left out."""
    if len(items) <= limit:
        return list(items), 0
    return list(items[:limit]), len(items) - limit


def evidence_items(details: dict[str, Any]) -> list[str]:
    """Return the judge's own ranked evidence entries in their reported order."""
    return [str(item) for item in details.get("evidence") or []]


def deduction_items(details: dict[str, Any]) -> list[str]:
    """Order quality deductions by the impact each states, largest deduction first."""
    deductions = [str(item) for item in details.get("deductions") or []]
    return sorted(deductions, key=lambda line: -_impact(line))


def _wrapped(items: list[str]) -> list[str]:
    """Wrap each chosen item into the display lines the panel prints."""
    return [line for item in items for line in wrap_by_width(item)]


def project_context(details: Any) -> dict[str, Any]:
    """Prepare one annotated project entry for the PDF report."""
    if not isinstance(details, dict):
        details = {"status": 500, "content": str(details)}
    status = details.get("status", 200)
    ai = details.get("ai_score")
    code = details.get("code_score")
    failed = status != 200 or (ai is None and code is None)
    ai_items, ai_more = capped(evidence_items(details), EVIDENCE_LIMIT)
    quality_items, quality_more = capped(deduction_items(details), DEDUCTION_LIMIT)
    context = {
        "ai": percent(ai),
        "code": percent(code),
        "failed": failed,
        "ai_items": _wrapped(ai_items),
        "ai_more": ai_more,
        "quality_items": _wrapped(quality_items),
        "quality_more": quality_more,
        "coverage": describe_coverage(details.get("coverage") or {}),
        "coverage_cell": coverage_cell(details.get("coverage") or {}),
        "failure": [],
    }
    if failed:
        reason = str(details.get("content", "")) or "The grader returned no report."
        # The opening lines only. The raw output can run to hundreds of characters and can
        # carry bytes echoed from a binary file; the full content remains in scores.json.
        context["failure"] = wrap_by_width(reason)[:FAILURE_LINES]
        if status != 200:
            context["status"] = f"Grader status {status}"
    return context


def build_context(scores: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert score data into contexts with display-ready project titles."""
    projects = []
    for name, details in scores.items():
        context = project_context(details)
        context["name"] = str(name)
        context["title_lines"] = title_lines(name)
        projects.append(context)
    return projects


def cover_context(
    source: Path | str | None,
    projects: list[dict[str, Any]],
    *,
    generated_at: str | None = None,
    model_tag: str | None = None,
) -> dict[str, Any]:
    """Prepare the cover page with run provenance and its project index."""
    stamp = generated_at or time.strftime("%Y-%m-%d %H:%M")
    source_name = Path(source).name if source else ""
    tag = grader_model_tag() if model_tag is None else model_tag
    provenance = [f"Generated {stamp}" + (f" from {source_name}" if source_name else "")]
    if tag:
        provenance.append(f"Grader model {tag}")
    provenance.append(f"{len(projects)} project(s)")
    rows = [
        {
            "name": str(project.get("name") or ""),
            "ai": str(project.get("ai", "n/a")),
            "code": str(project.get("code", "n/a")),
            "coverage": str(project.get("coverage_cell") or ""),
        }
        for project in projects
    ]
    return {
        "generated_at": stamp,
        "source": source_name,
        "model_tag": tag,
        "project_count": len(projects),
        "provenance": provenance,
        "rows": rows,
    }
