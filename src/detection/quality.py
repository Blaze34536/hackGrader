"""Build quality findings and format them as report-ready citations.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import re

from .metrics import _count, _file_at, _longest_function, _repeated_block, files_in
from .signals import (
    ABSENCE_DEDUCTIONS,
    COUNTED_PATTERNS,
    MISSING_README_REASON,
    QUALITY_DEDUCTIONS,
)

# The reader lists every path it read and every path it left out, as comma-separated paths, so
# a README named on one of those lines exists in the repository even though no excerpt of it
# reached the view.
_COVERAGE_PATHS = re.compile(
    r"(?im)^FILES (?:INCLUDED|NOT INCLUDED|NOT FETCHED) \(\d+\):[ \t]*(.*)$"
)


def _readme_named_in_coverage(code: str) -> bool:
    """Return whether the dump's own coverage headers name a README path."""
    return any(
        "readme" in entry.strip().lower()
        for match in _COVERAGE_PATHS.finditer(code)
        for entry in match.group(1).split(",")
    )


def quality_findings(code: str) -> list[dict]:
    """Collect fixed quality deductions and repository-scale maintainability signals."""
    findings = []
    text = [line for line in code.split("\n") if line.strip() and not line.startswith("--- FILE:")]
    for pattern, points, reason in ABSENCE_DEDUCTIONS:
        if pattern.search(code):
            continue
        if reason == MISSING_README_REASON and _readme_named_in_coverage(code):
            # Deducting here prices a file the reader chose not to include as though the
            # project had never written it. A read now spends its characters on code before
            # documentation, so a repository whose README was left out is the common case
            # rather than the rare one.
            continue
        findings.append({"points": points, "reason": reason, "file": None})
    for pattern, points, reason in QUALITY_DEDUCTIONS:
        matches = list(pattern.finditer(code))
        if matches:
            findings.append({
                "points": points,
                "reason": reason,
                "count": len(matches),
                "file": _file_at(code, matches[0].start()),
            })
    commented = _count(COUNTED_PATTERNS["commented_out"], code)
    if commented >= 4:
        findings.append({
            "points": 6,
            "reason": f"{commented} lines of commented-out code left in",
            "file": None,
        })
    debug = _count(COUNTED_PATTERNS["debug_prints"], code)
    if debug >= 3:
        findings.append({
            "points": 5,
            "reason": f"{debug} debug print statements left in the code",
            "file": None,
        })
    longest, name = _longest_function(code)
    if longest > 80:
        findings.append({
            "points": 6,
            "reason": f"a single function runs for {longest} lines (`{name}`)",
            "file": None,
        })
    if len(text) > 150 and not _count(COUNTED_PATTERNS["docs_comments"], code):
        findings.append({
            "points": 6,
            "reason": f"none of {len(text)} lines of code carry a comment",
            "file": None,
        })
    repeated = _repeated_block(code)
    if repeated:
        findings.append({
            "points": 5,
            "reason": f"a {repeated}-line block is copy-pasted at least four times",
            "file": None,
        })
    if len(text) < 15:
        findings.append({"points": 25, "reason": f"only {len(text)} lines of code were submitted", "file": None})
    if len(text) > 400 and len(files_in(code)) <= 2:
        findings.append({
            "points": 5,
            "reason": f"{len(text)} lines live in a single file with no structure",
            "file": None,
        })
    return findings


def ai_citation(finding: dict) -> str:
    """Format an authorship finding with its source quote when available."""
    if not finding["evidence"]:
        return finding["reason"]
    return f"{finding['reason']} - `{finding['evidence']}` in {finding['file']}"


def human_citation(finding: dict) -> str:
    """Format a human-trace finding with its count and first file."""
    return f"{finding['reason']} ({finding['count']}x; first seen in {finding['file']})"


def quality_citation(finding: dict) -> str:
    """Format a quality deduction with its score impact and source location."""
    suffix = f" ({finding['count']}x)" if finding.get("count", 0) > 2 else ""
    location = f" in {finding['file']}" if finding.get("file") else ""
    return f"-{finding['points']}% {finding['reason']}{suffix}{location}"
