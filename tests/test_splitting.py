"""Offline checks that source-disjoint holdouts mirror the pool they are drawn from.

Author: Nihal M
Date: 2026
License: MIT
"""

import os
import random
import tempfile
import unittest
from unittest import mock

from data.splitting import capacity, components, progress
from data.splitting.core import plan_partitions
from data.splitting.exclusion import exclude_repositories
from data.validation import MAX_TARGET_SHARE


class _Contract:
    """Read a bare integer target, which is all the splitter needs from a report."""

    @staticmethod
    def parse_report(text: str) -> dict:
        """Return the target score written as the whole assistant message."""
        return {"ai_score": int(text)}


def _repository(name: str, scores: list[int]) -> list[dict]:
    """Return one repository's evidence views, one row per target score."""
    url = f"https://github.com/acme/{name}"
    return [
        {
            "messages": [{}, {"content": f"{url} view {index}"}, {"content": str(score)}],
            "source_url": url,
            "source_type": "real_repository",
            "source_groups": [f"repository:{name}"],
        }
        for index, score in enumerate(scores)
    ]


def _plan(rows: list[dict]):
    """Plan a split of real-repository rows with small, reachable quotas."""
    # Only real repositories here: the configured family floors are not under test.
    with mock.patch.object(components, "validate_row"), \
            mock.patch.object(capacity, "REQUIRED_SOURCE_TYPES", {"real_repository"}), \
            mock.patch.object(
                capacity, "REQUIRED_SPLIT_SOURCE_TYPES", {"real_repository"}
            ), mock.patch.object(
                progress, "REQUIRED_SPLIT_SOURCE_TYPES", {"real_repository"}
            ):
        return plan_partitions(
            rows,
            (("validation", 0.15), ("test", 0.15)),
            random.Random(7),
            report_contract=_Contract,
            min_real_repositories_per_split=3,
            min_score_band_rows=3,
            min_middle_score_band_rows=0,
            min_rows=0,
            min_class_rows=5,
        )


class TestStratifiedHoldouts(unittest.TestCase):
    """Holdouts take each kind of repository in proportion, not the quota-richest first."""

    def test_holdouts_mirror_the_pool_mix(self) -> None:
        """Keep mostly-AI repositories near their pool share in every partition."""
        rows: list[dict] = []
        for index in range(40):
            # A mostly-AI repository's views span both classes, so it pays off more quotas.
            rows += _repository(f"ai{index}", [100, 100, 90, 0])
            rows += _repository(f"human{index}", [0, 0, 5, 10])
        plan = _plan(rows)
        placed: dict[str, list[bool]] = {"training": [], "validation": [], "test": []}
        for index, profile in enumerate(plan.profiles):
            mostly_ai = any("/ai" in url for url in profile.repositories)
            placed[plan.assigned.get(index, "training")].append(mostly_ai)
        for name, flags in placed.items():
            with self.subTest(partition=name):
                share = sum(flags) / len(flags)
                self.assertGreaterEqual(share, 0.4, f"{name}: {sum(flags)}/{len(flags)}")
                self.assertLessEqual(share, 0.6, f"{name}: {sum(flags)}/{len(flags)}")


class TestEqualHoldouts(unittest.TestCase):
    """Holdouts take equal thirds from human, partly-AI, and mostly-AI rows."""

    def test_unequal_groups_give_near_equal_holdout_thirds(self) -> None:
        """Draw near-equal prompts from each group despite a 1:3:1 pool."""
        rows: list[dict] = []
        for index in range(20):
            rows += _repository(f"human{index}", [0, 0, 0, 0])
            rows += _repository(f"ai{index}", [90, 90, 90, 90])
        for index in range(60):
            rows += _repository(f"mid{index}", [20, 20, 20, 20])
        plan = _plan(rows)
        for name in ("validation", "test"):
            counts = {"human": 0, "mid": 0, "ai": 0}
            for index, profile in enumerate(plan.profiles):
                if plan.assigned.get(index) == name:
                    group = next(iter(profile.repositories)).rsplit("/", 1)[1].rstrip("0123456789")
                    counts[group] += profile.prompt_count
            with self.subTest(partition=name):
                self.assertLessEqual(abs(counts["mid"] - counts["ai"]), 4, counts)
                self.assertGreaterEqual(counts["human"], counts["mid"] - 8, counts)
                # Every human row targets 0%, so its third must fit under the one-score cap.
                self.assertLessEqual(counts["human"] / sum(counts.values()), MAX_TARGET_SHARE)


class TestBenchmarkExclusion(unittest.TestCase):
    """Benchmark repositories never reach any partition."""

    def test_listed_repositories_are_dropped_by_canonical_url(self) -> None:
        """Match a listed repository however its URL is spelled."""
        rows = _repository("Bench", [0, 0]) + _repository("kept", [90])
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "benchmark.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("# held out\nhttps://github.com/ACME/bench.git/\n")
            kept = exclude_repositories(rows, path)
        self.assertEqual({row["source_url"] for row in kept}, {"https://github.com/acme/kept"})

    def test_no_list_keeps_every_row(self) -> None:
        """Leave the pool alone when exclusion is turned off."""
        rows = _repository("bench", [0])
        self.assertEqual(exclude_repositories(rows, ""), rows)


if __name__ == "__main__":
    unittest.main()
