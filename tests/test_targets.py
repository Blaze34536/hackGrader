"""Offline checks on the score targets the training data is built from.

The targets decide what the served judge learned to say, so the rules the prompt states are
also checked here: nothing that packages or describes a project may carry a sign, formatting
is not one, and an example the measurements place differently gets a different number.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import unittest

import contracts as report
import detection as detector
from data import config
from data.evidence import label_evidence, signal_line
from data.features import evidence_score, machine_likeness, visible_features
from data.labels import (
    LABEL_AI,
    LABEL_HUMAN,
    measured_share,
    provenance_target,
    validate_entries,
)
from data.rendering import batch_measurement

# One weighting for the checks below, so a target's arithmetic is read rather than inferred
# from feature fitting: documentation and comment density are the code-side measurements a
# repository example can carry.
WEIGHTS = {"documented functions": 0.5, "comment density": 0.5}

# Saturated documentation and explanatory comments, no trace of a person's habits: every
# measurement in WEIGHTS reads as fully machine-like.
DOCUMENTED_PROBE = "--- FILE: src/service.py ---\n" + "".join(
    f'def task_{index}(value):\n'
    f'    """task {index} value."""\n'
    f"    # step {index} of the pipeline\n"
    f"    return value * 2\n\n"
    for index in range(8)
)

# The opposite reading: no documentation, no comments, and human traces on every line.
TRACED_PROBE = "--- FILE: src/legacy.py ---\n" + "".join(
    f"def task_{index}(value):\n"
    f'    note_{index} = "TODO"\n'
    f'    print("step {index}")\n'
    f"    return value * 2\n\n"
    for index in range(6)
)


def target_of(dump: str) -> int:
    """Return the target the shared function assigns to one evidence view."""
    return evidence_score(detector.analyze(dump), WEIGHTS)


class TestTargetSpread(unittest.TestCase):
    """Check that targets use the range of the band an example was placed in."""

    def test_a_fully_machine_shaped_example_reaches_the_top_of_the_ai_band(self) -> None:
        """Reach the far edge of the AI band, so the strongest reading is distinguishable."""
        self.assertEqual(target_of(DOCUMENTED_PROBE), config.AI_LABEL_BAND[1])

    def test_a_trace_carrying_example_reaches_the_bottom_of_the_human_band(self) -> None:
        """Reach the far edge of the human band, so the weakest reading is too."""
        self.assertEqual(target_of(TRACED_PROBE), config.HUMAN_LABEL_BAND[0])

    def test_two_projects_the_measurements_place_apart_score_apart(self) -> None:
        """Spread the two readings over most of the range between the bands.

        Mapping the raw position onto the band wasted most of it: a supervised AI example
        had to reach 0.6 and the reachable range began at 69, which is why nearly every
        repository row targeted the same high number.
        """
        self.assertGreaterEqual(target_of(DOCUMENTED_PROBE) - target_of(TRACED_PROBE), 60)


class TestAbsenceIsNotMachineEvidence(unittest.TestCase):
    """Check that a trace nobody observed is undecided rather than machine-written."""

    def test_observing_no_human_trace_is_undecided(self) -> None:
        """Read a project carrying no human trace as undecided, not as machine-like.

        Inverting the feature outright made the absence of a TODO or a debug print worth
        full machine-likeness, which is the inference the prompt forbids and which carried
        most of a repository target: a project whose only distinguishing feature was being
        tidy scored in the high band.
        """
        likeness = machine_likeness(detector.analyze(DOCUMENTED_PROBE)["metrics"])

        self.assertEqual(likeness["human traces"], 0.5)

    def test_a_trace_that_was_observed_reads_as_human(self) -> None:
        """Keep a counted trace fully on the human side of the midpoint."""
        likeness = machine_likeness(detector.analyze(TRACED_PROBE)["metrics"])

        self.assertEqual(likeness["human traces"], 0.0)

    def test_the_same_project_with_and_without_tabs_reads_alike(self) -> None:
        """Leave a project's position unchanged by the whitespace characters it uses."""
        spaces = "--- FILE: src/app.py ---\n" + "def add(a, b):\n    return a+b\n" * 8
        tabs = spaces.replace("    return", "\treturn")

        self.assertGreater(detector.analyze(tabs)["metrics"]["tab_lines"], 0)
        self.assertEqual(
            machine_likeness(detector.analyze(spaces)["metrics"]),
            machine_likeness(detector.analyze(tabs)["metrics"]),
        )



class TestFormattingIsNotAFeature(unittest.TestCase):
    """Check that indentation and whitespace cannot carry a target any more."""

    def test_formatting_is_not_a_configured_feature(self) -> None:
        """Keep the formatting measurements out of both saturation maps."""
        for name in ("tab indent", "trailing whitespace"):
            with self.subTest(name=name):
                self.assertNotIn(name, config.AI_LIKE_FEATURES)
                self.assertNotIn(name, config.HUMAN_LIKE_FEATURES)

    def test_a_tab_indented_project_gains_no_machine_likeness_from_it(self) -> None:
        """Project only measurements the contract still reads, never a style count.

        ``machine_likeness`` inverts the human-like map, so a project indented with spaces
        read as fully machine-like for its zero tab count and the served model reported
        "0 line(s) are tab indented" as its strongest sign.
        """
        metrics = detector.analyze(
            "--- FILE: src/app.py ---\n" + "def add(a, b):\n\treturn a+b\n" * 8
        )["metrics"]

        self.assertGreater(metrics["tab_lines"], 0)
        features = visible_features(metrics)
        self.assertNotIn("tab indent", features)
        self.assertNotIn("trailing whitespace", features)


class TestTargetEvidence(unittest.TestCase):
    """Check that a target's citation comes from code, never from the packaging."""

    # The description is reproduced first, so an unfiltered selector reaches it first.
    VIEW = (
        "REPOSITORY: owner/repo\nFILES FOUND: 2\nFILES FETCHED: 2\n"
        "FILES INCLUDED (2): README.md, src/app.py\nFILES TRUNCATED: no\n\n"
        "--- FILE: README.md ---\n# Project\n\n## Features\n\n- fast\n\n- small\n\n"
        "--- FILE: src/app.py ---\n"
        + "".join(
            f'def task_{index}(value):\n    """task {index} value."""\n'
            "    return value * 2\n\n"
            for index in range(6)
        )
    )

    def test_the_strongest_readable_measurement_leads_the_entry(self) -> None:
        """State the measurement the target rests on, drawn from the code's measurements."""
        entry = label_evidence(self.VIEW, 1, detector.analyze(self.VIEW), WEIGHTS)

        self.assertEqual(len(entry), 1)
        self.assertIn("6 of 6 definitions have docstrings", entry[0])
        self.assertIn("src/app.py", entry[0])
        self.assertNotIn("README", entry[0])

    def test_a_generated_citation_is_one_the_serving_check_accepts(self) -> None:
        """Keep a target citable against its own view, which is what validation requires."""
        entry = label_evidence(self.VIEW, 1, detector.analyze(self.VIEW), WEIGHTS)
        rendered = report.render_report(
            target_of(self.VIEW), entry, 80, [], "One module beside its own description."
        )

        self.assertEqual(report.verify_citations(rendered, self.VIEW), [])

    def test_a_quote_never_comes_from_the_project_description(self) -> None:
        """Quote a code file even when the description is what a selector reaches first."""
        quote, name = signal_line(self.VIEW, 1)

        self.assertEqual(name, "src/app.py")
        self.assertNotIn("# Project", quote)

    def test_a_view_showing_only_the_description_yields_no_target(self) -> None:
        """Refuse a target for a batch that offers nothing but the project's own account."""
        readme = "--- FILE: README.md ---\n# Project\n\nA tiny module.\n\n- fast\n"

        self.assertEqual(
            label_evidence(readme, 1, detector.analyze(readme), WEIGHTS), []
        )


class TestVerifiedTargetsAreMeasured(unittest.TestCase):
    """Check that a verified label teaches the share it measured, not a band constant.

    A repository whose agent-attributed lines are 4% of its code is a 4, and one at 96%
    is a 96. The bands were a way to force the style reading apart without knowing the
    answer; with a measured attribution there is an answer, and a midpoint would teach the
    model a number the record contradicts.
    """

    def test_a_measured_line_share_becomes_the_target(self) -> None:
        """Supervise the percentage the manifest measured, not the band midpoint."""
        entry = {"label": LABEL_AI, "attributed_lines": 30, "examined_lines": 100}

        self.assertEqual(
            provenance_target(LABEL_AI, detector.analyze(DOCUMENTED_PROBE), entry), 30
        )

    def test_a_file_share_is_read_when_no_line_counts_exist(self) -> None:
        """Fall back to files rather than to a band when only file counts were recorded."""
        entry = {"label": LABEL_AI, "attributed_files": 1, "examined_files": 4}

        self.assertEqual(
            provenance_target(LABEL_AI, detector.analyze(DOCUMENTED_PROBE), entry), 25
        )

    def test_a_contradictory_measurement_is_ignored(self) -> None:
        """Refuse to read a count that cannot be true, and say what the target rests on."""
        entry = {"label": LABEL_AI, "attributed_lines": 120, "examined_lines": 100}
        low, high = config.AI_LABEL_BAND

        self.assertIsNone(measured_share(entry))
        self.assertEqual(
            provenance_target(LABEL_AI, detector.analyze(DOCUMENTED_PROBE), entry),
            round((low + high) / 2),
        )

    def test_an_ai_label_that_measured_nothing_is_dropped(self) -> None:
        """Drop a row whose own record contradicts the class it was filed under."""
        entry = {"label": LABEL_AI, "attributed_lines": 0, "examined_lines": 40}

        self.assertIsNone(
            provenance_target(LABEL_AI, detector.analyze(DOCUMENTED_PROBE), entry)
        )

    def test_a_verified_human_repository_measures_zero(self) -> None:
        """Teach zero where the protocol established that no agent wrote any of it."""
        entry = {"label": LABEL_HUMAN, "attributed_lines": 0, "examined_lines": 90}

        self.assertEqual(
            provenance_target(LABEL_HUMAN, detector.analyze(TRACED_PROBE), entry), 0
        )

    def test_a_verified_human_repository_without_counts_teaches_its_protocol(self) -> None:
        """Teach zero for the class whose protocol says no agent wrote any of it."""
        self.assertEqual(
            provenance_target(
                LABEL_HUMAN, detector.analyze(TRACED_PROBE), {"label": LABEL_HUMAN}
            ),
            0,
        )

    def test_an_ai_label_without_a_measurement_falls_back_to_its_band(self) -> None:
        """Keep the coarse fallback where a manifest recorded a class and nothing else."""
        low, high = config.AI_LABEL_BAND

        self.assertEqual(
            provenance_target(LABEL_AI, detector.analyze(DOCUMENTED_PROBE), {"label": LABEL_AI}),
            round((low + high) / 2),
        )

    def test_a_batch_takes_the_share_of_the_files_it_shows(self) -> None:
        """Measure a batch over its own files, so one repository gives several targets."""
        entry = {
            "label": LABEL_AI,
            "attributed_lines": 30,
            "examined_lines": 130,
            "files": [
                {"path": "a.py", "ai_lines": 30, "examined_lines": 30},
                {"path": "b.py", "ai_lines": 0, "examined_lines": 100},
            ],
        }
        agent_batch = "--- FILE: a.py ---\nprint(1)\n"
        human_batch = "--- FILE: b.py ---\nprint(2)\n"

        self.assertEqual(
            batch_measurement(entry, agent_batch),
            {"attributed_lines": 30, "examined_lines": 30},
        )
        self.assertEqual(provenance_target(LABEL_AI, detector.analyze(agent_batch), batch_measurement(entry, agent_batch)), 100)
        self.assertIsNone(
            provenance_target(LABEL_AI, detector.analyze(human_batch), batch_measurement(entry, human_batch))
        )

    def test_a_line_that_is_only_a_path_is_never_quoted(self) -> None:
        """Quote the next code line rather than a path, which the citation check reads as a file."""
        view = "--- FILE: src/x.py ---\nnotes/TODO_list.example.txt\nvalue = 1\n"

        quote = signal_line(view, 0)

        self.assertIsNotNone(quote)
        self.assertNotIn("TODO_list", quote[0])

    def test_a_batch_with_no_counted_file_keeps_the_repository_share(self) -> None:
        """Fall back to the repository share when a batch shows nothing the manifest counted."""
        entry = {
            "label": LABEL_AI,
            "attributed_lines": 30,
            "examined_lines": 100,
            "files": [{"path": "a.py", "ai_lines": 30, "examined_lines": 30}],
        }
        unlisted = "--- FILE: other.py ---\nprint(3)\n"

        self.assertIsNone(batch_measurement(entry, unlisted))
        self.assertEqual(
            provenance_target(LABEL_AI, detector.analyze(unlisted), batch_measurement(entry, unlisted) or entry),
            30,
        )

    def test_a_measurement_survives_the_manifest_validator(self) -> None:
        """Carry the counts through the fail-closed standard rather than around it."""
        entry = {
            "url": "https://github.com/owner/repo",
            "label": LABEL_AI,
            "evidence": [
                {
                    "evidence_class": "agent_contribution",
                    "pull_request": "https://github.com/owner/repo/pull/1",
                    "commit": "0f1e2d3c4b5a69788796a5b4c3d2e1f001234567",
                }
            ],
            "audited_by": "cohort-harvester",
            "audited_at": "2026-10-18",
            "attributed_lines": 30,
            "examined_lines": 100,
        }

        accepted, rejected = validate_entries([entry])

        self.assertEqual(rejected, [])
        stored = next(iter(accepted.values()))
        self.assertEqual(measured_share(stored), 30.0)


if __name__ == "__main__":
    unittest.main()
