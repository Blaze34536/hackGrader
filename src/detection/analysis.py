"""Coordinate detector metrics, findings, scores, and repository coverage.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import re
from typing import Any

from contracts.prompts import FILES_TRUNCATED_HEADER

from .findings import ai_findings, human_findings
from .metrics import files_in, metrics, strip_boilerplate
from .quality import ai_citation, human_citation, quality_citation, quality_findings
from .scoring import score_ai, score_quality


def analyze(code: str) -> dict[str, Any]:
    """Compute authorship, quality, metric, and repository coverage results."""
    scanned = strip_boilerplate(code)
    values = metrics(scanned)
    ai = ai_findings(scanned, values)
    human = human_findings(scanned)
    quality = quality_findings(scanned)
    shown = re.search(r"^FILES INCLUDED \((\d+)\)", code, re.MULTILINE)
    found_match = re.search(r"(?m)^FILES FOUND:\s*(\d+)", code)
    fetched_match = re.search(r"(?m)^FILES FETCHED:\s*(\d+)", code)
    shown_count = int(shown.group(1)) if shown else len(files_in(code))
    return {
        "ai_score": score_ai(ai, human),
        "ai_findings": ai,
        "ai_evidence": [ai_citation(finding) for finding in ai],
        "human_findings": human,
        "human_evidence": [human_citation(finding) for finding in human],
        "quality_score": score_quality(quality),
        "quality_findings": quality,
        "quality_deductions": [quality_citation(finding) for finding in quality],
        "metrics": values,
        "coverage": {
            "files_shown": shown_count,
            "files_found": int(found_match.group(1)) if found_match else shown_count,
            "files_fetched": int(fetched_match.group(1)) if fetched_match else shown_count,
            "truncated": (
                "[... Remaining files omitted for context limits ...]" in code
                or bool(FILES_TRUNCATED_HEADER.search(code))
            ),
        },
    }


def prompt_context(code: str) -> dict[str, Any]:
    """Return factual prompt measurements without calculating a detector verdict."""
    shown = re.search(r"^FILES INCLUDED \((\d+)\)", code, re.MULTILINE)
    found_match = re.search(r"(?m)^FILES FOUND:\s*(\d+)", code)
    fetched_match = re.search(r"(?m)^FILES FETCHED:\s*(\d+)", code)
    shown_count = int(shown.group(1)) if shown else len(files_in(code))
    return {
        "metrics": metrics(strip_boilerplate(code)),
        "coverage": {
            "files_shown": shown_count,
            "files_found": int(found_match.group(1)) if found_match else shown_count,
            "files_fetched": int(fetched_match.group(1)) if fetched_match else shown_count,
            "truncated": (
                "[... Remaining files omitted for context limits ...]" in code
                or bool(FILES_TRUNCATED_HEADER.search(code))
            ),
        },
    }
