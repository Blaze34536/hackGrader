"""Offline checks for the band-balanced cohort: AI-share bands, quotas, and human search.

Author: Nihal M
Date: 2026
License: MIT
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
import urllib.parse
from unittest import mock

from data.harvest import cli, manifest
from data.harvest.bands import AI_HIGH, AI_MID, band_of
from data.harvest.client import GithubClient
from data.harvest.cohort import harvest
from data.harvest.human import search_human_candidates
from data.harvest.records import (
    AGENT_FILE,
    HUMAN_FILE,
    CohortStore,
    FileTruth,
    PullRequest,
    RepositoryTruth,
    read_cohort,
)
from data.labels import load_repository_labels

from .test_harvest import (
    AGENT_FILE_LINES,
    HUMAN_FILE_LINES,
    FakeGithub,
    client_for,
    commit,
    copilot_commit,
    repository_routes,
    search_route,
)

QUERY = "is:pr is:merged author:app/copilot-swe-agent merged:2026-09-01..2026-09-30"


def truth(url: str, ai_lines: int, human_lines: int) -> RepositoryTruth:
    """Return a harvested repository whose measured share is ai / (ai + human)."""
    files = [
        FileTruth("src/agent.py", AGENT_FILE, 1, additions=ai_lines, current_lines=ai_lines),
        FileTruth("src/human.py", HUMAN_FILE, 1, current_lines=human_lines),
    ]
    pull = PullRequest(
        url=f"{url}/pull/1", number=1, agent="copilot-swe-agent", author="Copilot",
        merged_at="2026-09-02T12:00:00Z", merge_commit="merge" + "0" * 35,
    )
    return RepositoryTruth(url=url, pull_request=pull, files=tuple(files))


def candidate_fake(changed_files: int) -> FakeGithub:
    routes, texts = repository_routes(
        histories={
            "src/agent.py": [copilot_commit("agent-sha")],
            "src/human.py": [commit("human-sha")],
        },
        texts={"src/agent.py": AGENT_FILE_LINES, "src/human.py": HUMAN_FILE_LINES},
        tree={"truncated": False, "tree": [
            {"path": "src/agent.py", "type": "blob", "size": 400},
            {"path": "src/human.py", "type": "blob", "size": 300},
        ]},
        pull={"number": 7, "html_url": "https://github.com/acme/tool/pull/7",
              "merged_at": "2026-09-02T12:00:00Z", "merge_commit_sha": "merge" + "0" * 35,
              "user": {"login": "Copilot"}, "changed_files": changed_files},
    )
    routes["/repos/acme/tool/commits/agent-sha"] = {
        "files": [{"filename": "src/agent.py", "additions": 200}]
    }
    routes.update([search_route(QUERY, {"total_count": 1, "items": [
        {"html_url": "https://github.com/acme/tool/pull/7",
         "repository_url": "https://api.github.com/repos/acme/tool",
         "number": 7, "user": {"login": "Copilot"},
         "pull_request": {"merged_at": "2026-09-02T12:00:00Z"}},
    ]})])
    return FakeGithub(routes, texts)


class TestBands(unittest.TestCase):
    """A measured AI share falls in the mid band, the high band, or neither."""

    def test_band_edges(self) -> None:
        """Place each share in its band, with the gaps in neither."""
        cases = {None: None, 0.0: None, 0.09: None, 0.10: AI_MID, 0.39: AI_MID,
                 0.40: None, 0.49: None, 0.50: AI_HIGH, 1.0: AI_HIGH}
        for fraction, band in cases.items():
            with self.subTest(fraction=fraction):
                self.assertEqual(band_of(fraction), band)


class TestBalancedManifest(unittest.TestCase):
    def _entry(self, url: str, ai: int, examined: int) -> dict:
        return {"url": url, "label": "ai", "attributed_lines": ai, "examined_lines": examined}

    def test_each_band_is_capped_and_gaps_are_dropped(self) -> None:
        entries = [
            self._entry("https://github.com/a/low", 3, 100),
            self._entry("https://github.com/a/mid1", 20, 100),
            self._entry("https://github.com/a/mid2", 30, 100),
            self._entry("https://github.com/a/gap", 45, 100),
            self._entry("https://github.com/a/high", 90, 100),
            {"url": "https://github.com/h/one", "label": "human"},
            {"url": "https://github.com/h/two", "label": "human"},
        ]
        kept, dropped = manifest.balance_entries(entries, goal=1)
        self.assertEqual(
            [entry["url"] for entry in kept],
            ["https://github.com/a/mid1", "https://github.com/a/high", "https://github.com/h/one"],
        )
        self.assertEqual(
            {url for url, _ in dropped},
            {"https://github.com/a/low", "https://github.com/a/mid2",
             "https://github.com/a/gap", "https://github.com/h/two"},
        )


class TestBandHarvest(unittest.TestCase):
    def _harvest(self, folder: str, recorded: list[RepositoryTruth], fake: FakeGithub):
        """Record repositories into a cohort, then harvest with a goal of one per band."""
        store = CohortStore(os.path.join(folder, "cohort.jsonl"))
        for item in recorded:
            store.add_repository(item)
        client, _ = client_for(fake)
        with contextlib.redirect_stdout(io.StringIO()):
            summary = harvest(client, since="2026-09-01", until="2026-09-30",
                              per_agent_limit=5, max_repositories=10, store=store,
                              band_goal=1)
        return summary, read_cohort(store.path)

    def test_a_candidate_is_harvested_while_any_band_is_open(self) -> None:
        fake = candidate_fake(changed_files=2)  # looks like the full high band
        with tempfile.TemporaryDirectory() as folder:
            _, cohort = self._harvest(
                folder, [truth("https://github.com/x/high", 90, 10)], fake
            )
        self.assertEqual(len(cohort.repositories), 2)

    def test_a_candidate_for_an_open_band_is_harvested(self) -> None:
        fake = candidate_fake(changed_files=2)
        with tempfile.TemporaryDirectory() as folder:
            _, cohort = self._harvest(folder, [], fake)
        self.assertEqual([item.url for item in cohort.repositories],
                         ["https://github.com/acme/tool"])

    def test_the_band_goal_replaces_the_repository_total(self) -> None:
        fake = candidate_fake(changed_files=2)
        recorded = [truth(f"https://github.com/x/low{index}", 1, 99) for index in range(10)]
        with tempfile.TemporaryDirectory() as folder:
            _, cohort = self._harvest(folder, recorded, fake)
        self.assertIn("https://github.com/acme/tool", [item.url for item in cohort.repositories])

    def test_the_harvest_stops_once_every_band_is_full(self) -> None:
        fake = candidate_fake(changed_files=2)
        recorded = [truth("https://github.com/x/mid", 20, 80),
                    truth("https://github.com/x/high", 90, 10)]
        with tempfile.TemporaryDirectory() as folder:
            summary, _ = self._harvest(folder, recorded, fake)
        self.assertIn("band goal", summary.stopped)
        self.assertEqual(fake.requests, [])


class TestHumanSearch(unittest.TestCase):
    def test_search_returns_repository_urls_pushed_before_the_cutoff(self) -> None:
        fake = FakeGithub({"/search/repositories": {"total_count": 2, "items": [
            {"html_url": "https://github.com/old/one", "fork": False},
            {"html_url": "https://github.com/old/two", "fork": False},
        ]}})
        client, _ = client_for(fake)
        urls = search_human_candidates(
            client, before="2023-01-01", limit=2, languages=("python",)
        )
        self.assertEqual(urls, ["https://github.com/old/one", "https://github.com/old/two"])
        query = urllib.parse.unquote(fake.urls_like("/search/repositories")[0])
        self.assertIn("pushed:<2023-01-01", query)
        self.assertIn("language:python", query)


class TestBenchmarkHoldout(unittest.TestCase):
    def test_a_benchmark_repository_found_by_search_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            benchmark = os.path.join(folder, "benchmark.txt")
            with open(benchmark, "w", encoding="utf-8") as handle:
                handle.write("https://github.com/antirez/smaz\n")
            store = CohortStore(os.path.join(folder, "cohort.jsonl"))
            args = mock.Mock(human_repositories=None, human_search=2, refresh=False,
                             human_before="2023-01-01", human_sample=100)
            found = ["https://github.com/antirez/smaz", "https://github.com/old/kept"]
            with mock.patch.object(cli, "BENCHMARK_REPOS_FILE", benchmark), \
                    mock.patch.object(cli, "search_human_candidates", return_value=found), \
                    mock.patch.object(
                        cli, "verify_human_repositories", return_value=([], [])
                    ) as verify, contextlib.redirect_stdout(io.StringIO()):
                cli._verify_humans(args, store, mock.Mock())
        self.assertEqual(verify.call_args.args[1], ["https://github.com/old/kept"])


class TestBandGoalCommand(unittest.TestCase):
    def test_the_manifest_keeps_only_in_band_repositories(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            cohort_path = os.path.join(folder, "cohort.jsonl")
            labels_path = os.path.join(folder, "labels.jsonl")
            store = CohortStore(cohort_path)
            store.add_repository(truth("https://github.com/x/low", 3, 97))
            store.add_repository(truth("https://github.com/x/high", 90, 10))
            fake = FakeGithub({})

            def client_from_budget(budget, **_kwargs):
                return GithubClient(budget, fetch=fake)

            with mock.patch.object(cli, "GithubClient", client_from_budget), \
                    mock.patch.object(cli, "github_token", lambda: "token"), \
                    contextlib.redirect_stdout(io.StringIO()):
                cli.main([
                    "--agents", "copilot-swe-agent", "--since", "2026-09-01",
                    "--until", "2026-09-30", "--out", cohort_path,
                    "--labels-out", labels_path, "--band-goal", "1",
                ])
            loaded = load_repository_labels(labels_path)
        self.assertEqual(loaded.label_for("https://github.com/x/high"), "ai")
        self.assertIsNone(loaded.label_for("https://github.com/x/low"))


if __name__ == "__main__":
    unittest.main()
