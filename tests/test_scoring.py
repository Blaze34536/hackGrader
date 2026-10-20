"""Offline repository-dump budget checks and live Ollama scoring integration tests.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import pytest

from contracts import MODEL_DUMP_CHARS, is_valid_report, parse_report, trim_dump
from grading.runtime import Model

from .tools.scoring import ARTIFACT, summary_table, write_artifact
from .tools.stats import benchmark_stats, benchmark_timing

REPOS = (
    "https://github.com/rexim/aoc-2023",
    "https://github.com/levz0r/claude-code-statusline"
)

class TestDumpWindow(unittest.TestCase):
    """Check that repository evidence fits the model's configured context budget."""

    def test_a_dump_that_already_fits_is_untouched(self) -> None:
        """Leave already bounded repository evidence unchanged."""
        dump = "--- FILE: app.py ---\nprint('hi')\n"
        self.assertEqual(trim_dump(dump), dump)

    def test_whole_files_are_preferred_before_a_file_is_cut_short(self) -> None:
        """Retain content from multiple files when they jointly exceed the budget."""
        first = "--- FILE: a.py ---\n" + "a" * (MODEL_DUMP_CHARS - 500)
        second = "--- FILE: b.py ---\n" + "b" * (MODEL_DUMP_CHARS - 500)

        trimmed = trim_dump(first + "\n" + second)

        self.assertLessEqual(len(trimmed), MODEL_DUMP_CHARS)
        self.assertIn("--- FILE: a.py ---", trimmed)
        self.assertIn("--- FILE: b.py ---", trimmed)

    def test_a_dump_of_only_oversized_files_is_not_left_as_a_bare_header(self) -> None:
        """Ensure oversized evidence includes source text rather than metadata alone."""
        header = "REPOSITORY: o/r\nFILES INCLUDED (2): big.py, other.py"
        dump = (
            header
            + "\n\n--- FILE: big.py ---\n"
            + "x" * (MODEL_DUMP_CHARS * 3)
            + "\n\n--- FILE: other.py ---\n"
            + "y" * (MODEL_DUMP_CHARS * 3)
        )

        trimmed = trim_dump(dump)

        self.assertEqual(len(trimmed), MODEL_DUMP_CHARS)
        self.assertIn("--- FILE: big.py ---", trimmed)

    def test_a_large_file_does_not_stop_smaller_ones_being_kept(self) -> None:
        """Preserve later small files after a large file consumes part of the budget."""
        tiny = "--- FILE: main.c ---\nint main(void) {}\n"
        huge = "--- FILE: generated.h ---\n" + "x" * (MODEL_DUMP_CHARS * 2) + "\n"
        small = "--- FILE: util.py ---\nprint('ok')\n"

        trimmed = trim_dump(f"{tiny}{huge}{small}")

        self.assertIn("--- FILE: main.c ---", trimmed)
        self.assertIn("--- FILE: util.py ---", trimmed)
        self.assertLessEqual(len(trimmed), MODEL_DUMP_CHARS)

    def test_a_trimmed_view_shows_code_before_the_readme(self) -> None:
        """Spend a cut view's characters on the code before the project's own README."""
        header = (
            "REPOSITORY: o/r\nFILES INCLUDED (3): README.md, src/app.py, util.py"
        )
        dump = (
            header
            + "\n\n--- FILE: README.md ---\n"
            + "documentation line\n" * 40
            + "\n\n--- FILE: src/app.py ---\n"
            + "s" * (MODEL_DUMP_CHARS * 2)
            + "\n\n--- FILE: util.py ---\n"
            + "u" * (MODEL_DUMP_CHARS * 2)
        )

        trimmed = trim_dump(dump)

        self.assertLessEqual(len(trimmed), MODEL_DUMP_CHARS)
        self.assertIn("--- FILE: README.md ---", trimmed)
        self.assertLess(
            trimmed.index("--- FILE: src/app.py ---"),
            trimmed.index("--- FILE: README.md ---"),
        )
        self.assertLess(
            trimmed.index("--- FILE: util.py ---"),
            trimmed.index("--- FILE: README.md ---"),
        )

    def test_one_small_file_does_not_waste_the_rest_of_the_budget(self) -> None:
        """Use spare context on file excerpts after small files have been included."""
        header = (
            "REPOSITORY: o/r\nFILES INCLUDED (3): README.md, doc/tiny.js, src/big.py"
        )
        dump = (
            header
            + "\n\n--- FILE: README.md ---\n"
            + "r" * (MODEL_DUMP_CHARS * 2)
            + "\n\n--- FILE: doc/tiny.js ---\nconsole.log(1)\n"
            + "\n\n--- FILE: src/big.py ---\n"
            + "s" * (MODEL_DUMP_CHARS * 2)
        )

        trimmed = trim_dump(dump)

        self.assertLessEqual(len(trimmed), MODEL_DUMP_CHARS)
        self.assertGreater(
            len(trimmed),
            MODEL_DUMP_CHARS - 200,
            "a file that fit left most of the window unused",
        )

    def test_one_oversized_file_is_cut_to_the_budget(self) -> None:
        """Bound a single-file dump and mark its partial excerpt."""
        dump = "--- FILE: huge.py ---\n" + "x" * (MODEL_DUMP_CHARS * 2)
        self.assertEqual(len(trim_dump(dump)), MODEL_DUMP_CHARS)

    def test_the_budget_fits_the_training_window(self) -> None:
        """Keep the serving character budget within the training token window."""
        self.assertLessEqual(MODEL_DUMP_CHARS, 12_400)


class TestGraderInterface(unittest.TestCase):
    """Check the grader members the rest of the pipeline calls on it."""

    def test_the_retry_wrapper_can_name_the_repository_it_is_retrying(self) -> None:
        """Expose the URL helper the retry wrapper asks its grader for."""
        self.assertEqual(
            Model.extract_url(f"please evaluate {REPOS[0]} for AI involvement"),
            REPOS[0],
        )
        self.assertEqual(Model.extract_url("evaluate the attached project"), "")


@pytest.mark.live
class TestScoring(unittest.TestCase):
    """Exercise the live judge and verify its report output."""
    @benchmark_stats
    @benchmark_timing
    def test_querying(self) -> None:
        """Grade sample repositories and verify judge reports."""
        grader: Model = Model()
        not_ready = grader.preflight_error()
        if not_ready:
            self.fail(
                f"Live scoring needs a served model, so nothing was scored: {not_ready} "
                "See tests/README.md."
            )
        reports: dict[str, str] = {}
        analyses: dict[str, dict] = {}
        try:
            self._score_every_repository(grader, reports, analyses)
        finally:
            # pytest hides a passing test's output unless -s is passed, so the conftest
            # session summary repeats these results, and the artifact holds a partial run's
            # scores when a later repository fails.
            if reports:
                print(summary_table(reports, analyses))
                print(f"Full reports and prompt context written to {ARTIFACT}")

        if len(set(reports.values())) < len(reports):
            self.skipTest(
                "The configured grader model returned identical text for two different "
                "repositories, so it is not reading them. Point HACKGRADER_MODEL at a model "
                "that grounds its answers in the code (see README) and rerun. The pipeline "
                "still finishes such a run, but identical reports should be reviewed manually."
            )

    def _score_every_repository(
        self, grader: Model, reports: dict[str, str], analyses: dict[str, dict]
    ) -> None:
        """Grade each sample repository, saving the reports as they finish."""
        for repo in REPOS:
            with self.subTest(repo=repo):
                question: str = (
                    "look at this repository and give it a percent numeric score based on "
                    f"how good the code is: {repo}"
                )
                answer: dict = grader.query_with_retry(question)
                print(answer["content"])

                self.assertEqual(answer["status"], 200, f"Query failure for {repo}")
                self.assertTrue(
                    is_valid_report(answer["content"]),
                    f"Unparseable report for {repo}: {answer['content'][:200]}",
                )

                parsed = parse_report(answer["content"])
                self.assertIsNotNone(parsed["ai_score"])
                self.assertIsNotNone(parsed["code_score"])

                analysis = answer.get("analysis") or {}
                reports[repo] = answer["content"]
                analyses[repo] = analysis
                # Save as each repository finishes so a later failure does not discard
                # already completed judge reports.
                write_artifact(ARTIFACT, reports, analyses)


class TestScoringOutput(unittest.TestCase):
    """Check the judge-only summary and the artifact a live run leaves behind."""

    REPORT = (
        "1. AI Involvement Score (0-100%): 12%\n"
        "Ranked AI evidence (strongest to weakest):\n"
        "Rank 1: `print` in app.py is a weak, non-specific sign.\n\n"
        "2. Code Quality Score (0-100%): 78%\n"
        "- Deductions: no deductions applied\n\n"
        "3. Summary & Notable Findings: A short script that prints one value from app.py.\n"
    )
    ANALYSIS: ClassVar[dict] = {
        "coverage": {"files_shown": 3, "files_found": 4, "files_fetched": 4},
    }

    def test_the_summary_shows_judge_scores_and_files_read(self) -> None:
        """Print judge scores and the files the run read, without a second score."""
        table = summary_table({"owner/repo": self.REPORT}, {"owner/repo": self.ANALYSIS})

        header, row = table.splitlines()
        self.assertNotIn("detector", header)
        self.assertEqual(row.split()[:4], ["owner/repo", "12%", "78%", "3"])
        self.assertEqual(row.split()[-1], "ok")

    def test_an_unparseable_report_is_marked_rather_than_scored(self) -> None:
        """Mark a report the parser cannot read instead of printing invented scores."""
        table = summary_table({"owner/broken": "no report here"}, {})

        self.assertIn("unparseable", table)
        self.assertIn("n/a", table)

    def test_the_artifact_round_trips_every_report(self) -> None:
        """Save the reports and prompt coverage so a finished run can be read back."""
        with tempfile.TemporaryDirectory() as directory:
            path = write_artifact(
                Path(directory) / "nested" / "last-run.json",
                {"owner/repo": self.REPORT},
                {"owner/repo": self.ANALYSIS},
            )
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertIn("written", saved)
        self.assertEqual(saved["repositories"]["owner/repo"]["report"], self.REPORT)
        self.assertEqual(
            saved["repositories"]["owner/repo"]["analysis"]["coverage"]["files_shown"], 3
        )
        self.assertIn("owner/repo", saved["summary"])


if __name__ == "__main__":
    unittest.main()
