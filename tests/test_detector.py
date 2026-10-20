"""Legacy detector regressions and shared report citation-contract tests.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import unittest

from contracts import (
    deduction_claims,
    is_valid_report,
    parse_report,
    render_report,
    verify_citations,
)
from contracts.repository_dump import _parse_metadata
from detection import analyze, build_report, prompt_context
from grading.reporting import Reporting
from pdf.layout import sanitize

from .fixtures.detector import FIXTURES, GENERATED, HAND_WRITTEN


class TestReportingAnnotation(unittest.TestCase):
    """Check that canonical report content is safely parsed and annotated."""

    def test_valid_report_content_is_parsed_without_an_analysis_block(self) -> None:
        """Retain parsed judge scores when no deterministic analysis accompanies a report."""
        content = render_report(
            20,
            ["observed `print` in app.py"],
            80,
            [],
            "The app entry point is compact and its output is straightforward to review.",
        )

        result = Reporting()._annotate({"status": 200, "content": content})

        self.assertEqual(result["ai_score"], 20)
        self.assertEqual(result["code_score"], 80)
        self.assertEqual(result["analysis"], {"metrics": {}, "coverage": {}})

    def test_non_mapping_result_is_converted_to_a_report_failure(self) -> None:
        """Turn unexpected result values into an error record without flags."""
        result = Reporting()._annotate("unexpected result")

        self.assertEqual(result["status"], 500)
        self.assertEqual(result["analysis"], {"metrics": {}, "coverage": {}})


class TestDetector(unittest.TestCase):
    """Check baseline scores, evidence attribution, and report round trips."""

    def setUp(self) -> None:
        """Prepare human and generated fixture results for detector regressions."""
        self.hand = analyze(HAND_WRITTEN)
        self.generated = analyze(GENERATED)
        self.threshold = 40

    def test_detector_remains_available_for_diagnostics_only(self) -> None:
        """Keep the deterministic detector as a diagnostic, not a project judge."""
        self.assertIsInstance(self.hand["ai_score"], int)
        self.assertIsInstance(self.generated["ai_score"], int)
        self.assertGreaterEqual(self.generated["ai_score"], self.hand["ai_score"])

    def test_prompt_context_contains_only_metrics_and_coverage(self) -> None:
        """Keep detector verdicts out of the contextual data supplied to the judge."""
        context = prompt_context(GENERATED)

        self.assertEqual(set(context), {"metrics", "coverage"})
        self.assertNotIn("ai_score", context)
        self.assertNotIn("quality_score", context)
        self.assertNotIn("ai_findings", context)

    def test_explicit_ai_project_claims_are_recognized(self) -> None:
        """Recognize explicit project-level AI-authorship declarations."""
        claims = (
            "This entire repository was created by AI.",
            "100% AI-generated code.",
            "Zero human-written code was added.",
            "Created through collaboration between a human and AI.",
        )
        for claim in claims:
            with self.subTest(claim=claim):
                analysis = analyze(
                    f"REPOSITORY: owner/project\nFILES INCLUDED (1): README.md\n\n"
                    f"--- FILE: README.md ---\n{claim}"
                )
                self.assertGreaterEqual(analysis["ai_score"], 40)
                self.assertTrue(
                    any(
                        "explicit AI-generation marker" in finding["reason"]
                        for finding in analysis["ai_findings"]
                    ),
                    analysis["ai_findings"],
                )

    def test_toolchain_boilerplate_is_not_an_ai_marker(self) -> None:
        """Avoid treating ordinary generated build metadata as an authorship marker."""
        unity = (
            "REPOSITORY: student/game\nDEFAULT BRANCH: main\n"
            "FILES INCLUDED (2): .gitignore, Assets/Scripts/Player.cs\n\n"
            "--- FILE: .gitignore ---\n# Temporary auto-generated Android Assets\n"
            "/[Aa]ssets/[Ss]treamingAssets/aa.meta\n# Mono auto generated files\n"
            "mono_crash.*\n\n--- FILE: Assets/Scripts/Player.cs ---\n"
            "using UnityEngine;\n\npublic class Player : MonoBehaviour\n{\n"
            "    void Update()\n    {\n"
            '        transform.Translate(Input.GetAxis("Horizontal"), 0, 0);\n'
            "    }\n}\n"
        )
        analysis = analyze(unity)
        self.assertFalse(
            any("explicit AI-generation marker" in text for text in analysis["ai_evidence"]),
            analysis["ai_evidence"],
        )
        self.assertIsInstance(analysis["ai_score"], int)

    def test_docstrings_that_restate_the_name_are_detected(self) -> None:
        """Detect uniformly repetitive function documentation as one signal."""
        reasons = [finding["reason"] for finding in self.generated["ai_findings"]]
        self.assertTrue(any("restate the function name" in reason for reason in reasons), reasons)
        metrics = self.generated["metrics"]
        self.assertEqual(metrics["restated_docstrings"], metrics["docstrings"])

    def test_documentation_volume_moves_the_score_across_a_graded_range(self) -> None:
        """Treat documentation volume as a graded signal once the file is large enough to judge."""
        plain_head = '''--- FILE: a.py ---
import os

def load(name):
    return open(os.path.join("d", name)).read()

def save(name, body):
    return open(os.path.join("d", name), "w").write(body)
'''
        documented_head = '''--- FILE: a.py ---
import os

# ============ Notes ============

def load(name: str) -> str:
    """Load name."""
    return open(os.path.join("d", name)).read()

def save(name: str, body: str) -> int:
    """Save name."""
    return open(os.path.join("d", name), "w").write(body)
'''
        plain_filler = "".join(
            f"def step_{index}(value):\n    return value + {index}\n\n" for index in range(16)
        )
        documented_filler = "".join(
            f'def step_{index}(value: int) -> int:\n    """Step {index}."""\n'
            f"    return value + {index}\n\n"
            for index in range(16)
        )
        bare = analyze(plain_head + plain_filler)["ai_score"]
        shaped = analyze(documented_head + documented_filler)["ai_score"]
        self.assertLess(bare, self.threshold)
        self.assertGreaterEqual(shaped, 40)
        self.assertGreater(shaped - bare, 25)

    def test_tiny_files_do_not_score_documentation_or_cleanliness(self) -> None:
        """Keep comment and docstring ratios from deciding authorship for a few lines of code."""
        tiny = (
            "--- FILE: a.py ---\n"
            "# read the file\n"
            "def load(name: str) -> str:\n"
            '    """Load name."""\n'
            "    return name\n"
        )
        analysis = analyze(tiny)
        self.assertIsInstance(analysis["ai_score"], int)
        shape_families = {"restatement", "annotations", "comments", "cleanliness"}
        self.assertFalse(
            [finding for finding in analysis["ai_findings"] if finding["family"] in shape_families],
            analysis["ai_findings"],
        )

    def test_every_finding_names_a_file_it_read(self) -> None:
        """Attribute evidence findings to files present in the dump."""
        for analysis, label in ((self.hand, "hand"), (self.generated, "generated")):
            for finding in analysis["ai_findings"]:
                if finding["evidence"]:
                    self.assertIn(finding["file"], ("app.py", "helpers.py"), label)

    def test_a_readme_the_reader_left_out_is_not_a_missing_readme(self) -> None:
        """Price a repository with no README, not one whose README was not included."""
        header = (
            "REPOSITORY: owner/repo\nDEFAULT BRANCH: main\nFILES FOUND: 3\n"
            "FILES FETCHED: 3\nFILES INCLUDED (1): src/app.py\nFILES TRUNCATED: yes\n"
        )
        source = "\n--- FILE: src/app.py ---\n" + "print('value')\n" * 30
        omitted = header + "FILES NOT INCLUDED (2): README.md, docs/guide.md\n" + source
        absent = header + source

        left_out = deduction_claims(build_report(omitted, seed=3))
        self.assertFalse(
            any("readme" in claim.lower() for claim in left_out), left_out
        )
        missing = deduction_claims(build_report(absent, seed=3))
        self.assertTrue(any("readme" in claim.lower() for claim in missing), missing)

    def test_target_citations_appear_in_the_dump(self) -> None:
        """Verify generated report evidence quotes source text that was actually shown."""
        for dump, label in FIXTURES:
            name, _, body = _parse_metadata(dump).files[0]
            line = next(text.strip() for text in body.splitlines() if text.strip())
            target = build_report(
                dump,
                seed=3,
                ai_evidence=[
                    f"the source contains this line, for example `{line}` in {name}"
                ],
            )
            self.assertEqual(verify_citations(target, dump), [], label)

    def test_measurement_prose_without_a_quote_is_not_a_citation(self) -> None:
        """Refuse an evidence line that quotes nothing: it cannot be checked against a view."""
        for dump, label in FIXTURES:
            self.assertTrue(verify_citations(build_report(dump, seed=3), dump), label)

    def test_targets_round_trip_through_the_runtime_parser(self) -> None:
        """Keep deterministic reports compatible with the serving parser."""
        for dump, label in FIXTURES:
            target = build_report(dump, seed=3)
            parsed = parse_report(target)
            self.assertTrue(is_valid_report(target), target)
            self.assertIsNotNone(parsed["ai_score"], label)
            self.assertIsNotNone(parsed["code_score"], label)
            self.assertGreaterEqual(len(parsed["summary"]), 20, label)

    def test_target_summary_stays_one_short_sentence(self) -> None:
        """Keep the target narrative as brief as the shared prompt asks the judge to be."""
        for dump, label in FIXTURES:
            summary = parse_report(build_report(dump, seed=3))["summary"]
            self.assertLessEqual(len(summary.split()), 30, f"{label}: {summary}")
            self.assertRegex(summary, r"`[^`]+`", label)


class TestPdfText(unittest.TestCase):
    """Check that the PDF text fold keeps prose punctuation readers rely on."""

    def test_en_and_em_dashes_are_kept(self) -> None:
        """Keep dashes in review prose instead of folding them to hyphens."""
        self.assertEqual(
            sanitize("review \u2014 notes \u2013 done"),
            "review \u2014 notes \u2013 done",
        )


if __name__ == "__main__":
    unittest.main()
