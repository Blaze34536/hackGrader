"""Build explainable authorship findings from source and documentation signals.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from typing import Any

from .base import (
    CLEANLINESS_POINTS,
    COMMENT_DENSITY_TARGET,
    COMMENT_POINTS,
    CORROBORATION_CAP,
    CORROBORATION_POINTS,
    MIN_DOCUMENTED_FUNCTIONS,
    MIN_SCORED_SOURCE_LINES,
    README_IDEAL_CHARS,
    README_MIN_CHARS,
    README_POINTS,
    README_STRUCTURE_IDEAL,
    RESTATEMENT_FLOOR,
    RESTATEMENT_POINTS,
    SATURATION_FLOOR,
    TRACE_LIMIT,
    TRACES,
    VOLUME_IDEAL_LINES,
)
from .metrics import _file_at, _in_documentation
from .signals import AI_SIGNALS, HUMAN_SIGNALS


def _finding(
    weight: int, reason: str, evidence: str, code: str, offset: int, family: str = ""
) -> dict[str, Any]:
    """Create one AI finding with source location and signal family."""
    return {
        "weight": weight,
        "reason": reason,
        "evidence": evidence,
        "file": _file_at(code, offset),
        "family": family,
    }


def _corroboration_findings(findings: list[dict]) -> list[dict]:
    """Add a capped bonus when independent non-explicit signals agree."""
    families = {
        finding["family"]
        for finding in findings
        if finding["family"] and finding["family"] != "explicit"
    }
    bonus = min(CORROBORATION_POINTS * max(len(families) - 1, 0), CORROBORATION_CAP)
    if not bonus:
        return []
    return [
        _finding(
            bonus,
            f"{len(families)} independent signal families agree ({', '.join(sorted(families))})",
            "",
            "",
            0,
            "corroboration",
        )
    ]


def _readme_findings(code: str, values: dict) -> list[dict]:
    """Score README volume and structure while discounting first-person writing."""
    chars = values["readme_chars"]
    if chars < README_MIN_CHARS:
        return []
    structure = min(
        (values["readme_headings"] + values["readme_bullets"]) / README_STRUCTURE_IDEAL,
        1.0,
    )
    volume = min(chars / README_IDEAL_CHARS, 1.0)
    informal = min(values["readme_first_person"] / 8, 1.0)
    weight = round(
        README_POINTS
        * (0.5 * structure + 0.5 * volume)
        * (1 - 0.5 * informal)
        * (1 - 0.5 * min(values["readme_casual"] / 3, 1.0))
        * (0.5 if values["readme_typos"] else 1.0)
    )
    if not weight:
        return []
    reason = (
        f"the README runs {chars} characters in {values['readme_headings']} heading(s) "
        f"and {values['readme_bullets']} bullet(s) with {values['readme_first_person']} "
        "first-person reference(s)"
    )
    return [_finding(weight, reason, "", code, 0, "readme")]


def _documentation_findings(code: str, values: dict) -> list[dict]:
    """Score docstring restatement, annotations, and explanatory comment density."""
    findings: list[dict] = []
    functions = values["functions"]
    if functions >= MIN_DOCUMENTED_FUNCTIONS and values["docstrings"]:
        ratio = values["restated_docstrings"] / values["docstrings"]
        restatement = max(
            0.0, (ratio - RESTATEMENT_FLOOR) / (1 - RESTATEMENT_FLOOR)
        )
        if restatement:
            findings.append(
                _finding(
                    round(RESTATEMENT_POINTS * restatement),
                    f"{values['restated_docstrings']} of {values['docstrings']} docstrings just restate the function name",
                    "",
                    code,
                    0,
                    "restatement",
                )
            )
    if functions >= MIN_DOCUMENTED_FUNCTIONS:
        annotations = min(values["typed_functions"] / functions, 1.0)
        annotations = max(0.0, (annotations - SATURATION_FLOOR) / (1 - SATURATION_FLOOR))
        if annotations:
            findings.append(
                _finding(
                    round(6 * annotations),
                    f"{values['typed_functions']} of {functions} functions are fully type-annotated",
                    "",
                    code,
                    0,
                    "annotations",
                )
            )
    density = values["comment_lines"] / (values["code_lines"] or 1)
    points = round(COMMENT_POINTS * min(density / COMMENT_DENSITY_TARGET, 1.0))
    if points:
        findings.append(
            _finding(
                points,
                f"{values['comment_lines']} of {values['code_lines']} source line(s) are explanatory comments",
                "",
                code,
                0,
                "comments",
            )
        )
    return [finding for finding in findings if finding["weight"]]


def _shape_findings(code: str, values: dict) -> list[dict]:
    """Credit sufficiently large clean source samples for consistent structure."""
    volume = min(
        max(values["source_lines"] / VOLUME_IDEAL_LINES - SATURATION_FLOOR, 0.0)
        / (1 - SATURATION_FLOOR),
        1.0,
    )
    traces = sum(values[name] for name in TRACES)
    weight = round(CLEANLINESS_POINTS * (1 - min(traces / TRACE_LIMIT, 1.0)) * volume)
    if not weight:
        return []
    reason = (
        f"no TODO, debug print, commented-out line, local path or emoji in "
        f"{values['source_lines']} line(s) of source"
    )
    return [_finding(weight, reason, "", code, 0, "cleanliness")]


def ai_findings(code: str, values: dict) -> list[dict]:
    """Identify explicit markers and corroborating structural authorship evidence."""
    found: list[dict] = []
    for pattern, weight, reason, allow_documentation in AI_SIGNALS:
        matches = [
            match
            for match in pattern.finditer(code)
            if allow_documentation or not _in_documentation(code, match.start())
        ]
        if matches:
            evidence = matches[0].group(0).strip().splitlines()[0].strip()[:60]
            found.append(_finding(weight, reason, evidence, code, matches[0].start(), "explicit"))
    found.extend(_readme_findings(code, values))
    if values["source_lines"] >= MIN_SCORED_SOURCE_LINES:
        found.extend(_documentation_findings(code, values))
        found.extend(_shape_findings(code, values))
    found.extend(_corroboration_findings(found))
    return found


def human_findings(code: str) -> list[dict]:
    """Identify human-trace features and their first source file."""
    findings = []
    for pattern, weight, reason in HUMAN_SIGNALS:
        matches = list(pattern.finditer(code))
        if matches:
            findings.append(
                {
                    "weight": weight,
                    "reason": reason,
                    "count": len(matches),
                    "file": _file_at(code, matches[0].start()),
                }
            )
    return findings
