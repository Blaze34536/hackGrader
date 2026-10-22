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


class TestStratifiedHoldouts(unittest.TestCase):
    """Holdouts take each kind of repository in proportion, not the quota-richest first."""

    def test_holdouts_mirror_the_pool_mix(self) -> None:
        """Keep mostly-AI repositories near their pool share in every partition."""
        rows: list[dict] = []
        for index in range(40):
            # A mostly-AI repository's views span both classes, so it pays off more quotas.
            rows += _repository(f"ai{index}", [100, 100, 90, 0])
            rows += _repository(f"human{index}", [0, 0, 5, 10])
        # Only real repositories here: the configured family floors are not under test.
        with mock.patch.object(components, "validate_row"), \
                mock.patch.object(capacity, "REQUIRED_SOURCE_TYPES", {"real_repository"}), \
                mock.patch.object(
                    capacity, "REQUIRED_SPLIT_SOURCE_TYPES", {"real_repository"}
                ), mock.patch.object(
                    progress, "REQUIRED_SPLIT_SOURCE_TYPES", {"real_repository"}
                ):
            plan = plan_partitions(
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
        placed: dict[str, list[bool]] = {"training": [], "validation": [], "test": []}
        for index, profile in enumerate(plan.profiles):
            mostly_ai = any("/ai" in url for url in profile.repositories)
            placed[plan.assigned.get(index, "training")].append(mostly_ai)
        for name, flags in placed.items():
            with self.subTest(partition=name):
                share = sum(flags) / len(flags)
                self.assertGreaterEqual(share, 0.4, f"{name}: {sum(flags)}/{len(flags)}")
                self.assertLessEqual(share, 0.6, f"{name}: {sum(flags)}/{len(flags)}")


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
