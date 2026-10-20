"""Annotate and persist judge results.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from contracts import authorship_evidence, deduction_claims, parse_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORES_PATH = str(PROJECT_ROOT / "scores.json")


class Reporting:
    """Enrich saved evaluation results with the judge's scores and evidence."""

    def __init__(self, json_path: str | None = None) -> None:
        """Choose an explicit scores file or the project-default path."""
        self.json_path = json_path or DEFAULT_SCORES_PATH

    def generate_report(self) -> None:
        """Parse judge reports, sort results by score, and persist them."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Error: The file {self.json_path} was not found.")
        with open(self.json_path, encoding="utf-8") as file:
            try:
                projects = json.load(file)
            except json.JSONDecodeError as exc:
                raise ValueError("Error: Invalid JSON format.") from exc

        annotated = self.annotate_all(projects)
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump(annotated, file, ensure_ascii=False, indent=4)
        print(f"Report generated for {len(annotated)} project(s). Written to {self.json_path}.")

    def annotate_all(self, projects: dict[str, Any]) -> dict[str, Any]:
        """Parse every stored judge report and sort by the lowest available score."""
        annotated = {name: self._annotate(project) for name, project in projects.items()}
        return dict(sorted(annotated.items(), key=lambda item: self._sort_key(item[1])))

    @staticmethod
    def _sort_key(project: dict) -> tuple[int, int]:
        """Sort projects by their weakest available judge score, with failures first."""
        scores = [
            score
            for score in (project.get("ai_score"), project.get("code_score"))
            if isinstance(score, int)
        ]
        return (0, min(scores)) if scores else (1, 0)

    @staticmethod
    def _annotate(project: Any) -> dict:
        """Add the judge's parsed scores, evidence, deductions, and prompt context."""
        if not isinstance(project, dict):
            project = {"status": 500, "content": str(project)}
        content = str(project.get("content", ""))
        parsed = parse_report(content)
        prior_analysis = project.get("analysis") or {}
        coverage = prior_analysis.get("coverage") or {}
        metrics = prior_analysis.get("metrics") or {}
        # Retain only factual prompt context from older runs; detector verdicts/findings
        # must not be mistaken for evaluation results by downstream consumers.
        project["analysis"] = {"metrics": metrics, "coverage": coverage}
        for obsolete in (
            "flags",
            "report_findings",
            "detector_ai_score",
            "detector_code_score",
            "detector_ai_findings",
            "detector_quality_findings",
            "human_evidence",
        ):
            project.pop(obsolete, None)
        project.update({
            "ai_score": parsed["ai_score"],
            "code_score": parsed["code_score"],
            "summary": parsed["summary"],
            "metrics": metrics,
            "evidence": authorship_evidence(content),
            "deductions": deduction_claims(content),
            "coverage": coverage,
            "model_citations": parsed["citations"],
        })
        return project
