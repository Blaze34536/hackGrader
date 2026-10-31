"""Offline checks that every team gets its own PDF, written as soon as it is graded.

Author: Nihal M
Date: 2026
License: MIT
"""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from grading.reporting import Reporting
from hackgrader.pipeline import workflow
from pdf import cli

from . import test_report as fixtures

FIRST = "https://github.com/student/first"
SECOND = "https://github.com/student/second"


class TestTeamReports(unittest.TestCase):
    """The PDF command writes one report per team into a reports folder."""

    def test_each_team_gets_a_pdf_named_after_its_repository(self) -> None:
        scores = {
            FIRST: {"status": 200, "content": fixtures.TestSavedScoresFile.CLEAN_REPORT},
            SECOND: fixtures.TestJudgeOnlyRendering._stale()[fixtures.TestJudgeOnlyRendering.PROJECT],
        }
        with TemporaryDirectory() as directory:
            source = Path(directory) / "scores.json"
            source.write_text(json.dumps(scores), encoding="utf-8")
            reports = Path(directory) / "reports"

            self.assertEqual(cli.main([str(source), str(reports)]), 0)

            self.assertEqual(
                sorted(path.name for path in reports.iterdir()),
                ["student-first.pdf", "student-second.pdf"],
            )


class TestSequentialReports(unittest.TestCase):
    """The gallery run writes each team's PDF before it grades the next team."""

    def test_a_team_report_exists_before_the_next_team_is_graded(self) -> None:
        with TemporaryDirectory() as directory:
            reports = Path(directory) / "reports"
            seen: list[bool] = []

            def grade(_question: str) -> dict:
                seen.append((reports / "student-first.pdf").exists())
                return {"status": 200, "content": fixtures.TestSavedScoresFile.CLEAN_REPORT}

            crawler = mock.Mock()
            crawler.get_projects.return_value = ["p1", "p2"]
            crawler.get_github_projects.return_value = [FIRST, SECOND]
            grader = mock.Mock(query_with_retry=grade)
            grader.preflight_error.return_value = None
            scores_path = os.path.join(directory, "scores.json")
            with mock.patch.object(workflow, "Driver"), \
                    mock.patch.object(workflow, "Crawler", return_value=crawler), \
                    mock.patch.object(workflow, "Model", return_value=grader), \
                    mock.patch.object(
                        workflow, "Reporting", lambda: Reporting(json_path=scores_path)
                    ), \
                    mock.patch.object(workflow, "REPORTS_DIR", reports), \
                    mock.patch("builtins.print"):
                workflow.run_gallery("https://example.devpost.com")

            self.assertEqual(seen, [False, True])
            self.assertTrue((reports / "student-second.pdf").exists())


if __name__ == "__main__":
    unittest.main()
