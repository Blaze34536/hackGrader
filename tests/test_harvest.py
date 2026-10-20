"""Offline checks on the cohort harvest.

The harvest is what replaces a style reading with a record somebody else can recheck, so these
tests pin the parts that carry that claim: the last-touch rule and its bounds, the human
class's no-agent window, the request budget that stops a run cleanly, resumption from an
appended cohort, and the derived labels manifest being accepted by the same fail-closed
standard the data builder applies.

Every offline test runs against a recorded GitHub, injected as a plain function, so no test
reaches the network and each one can assert exactly which requests the harvest chose to spend.
The live class at the end is the exception and is deselected by default: it asks GitHub
directly whether the registered agent apps are still real, which no recording can prove.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
import urllib.parse
from typing import Any
from unittest import mock

import pytest

from data.harvest import cli, manifest
from data.harvest.agents import (
    AGENT_APPS,
    AGENT_SLUGS,
    AgentApp,
    agent_for_slug,
    agent_of,
)
from data.harvest.attribution import classify_touches, file_truth
from data.harvest.budget import BudgetExceeded, RequestBudget
from data.harvest.client import GithubClient, GithubError, github_token
from data.harvest.cohort import (
    candidate_files,
    harvest,
    harvest_repository,
    rank_candidates,
)
from data.harvest.human import verify_human_repository
from data.harvest.queries import (
    MergeCandidate,
    halve,
    month_windows,
    search_merged,
)
from data.harvest.records import (
    AGENT_FILE,
    HUMAN_FILE,
    UNKNOWN_FILE,
    CohortStore,
    HumanVerification,
    read_cohort,
)
from data.labels import load_repository_labels, require_usable, validate_entries

RATE_HEADERS = {"X-RateLimit-Limit": "5000", "X-RateLimit-Remaining": "4999"}


class FakeGithub:
    """A recorded GitHub API: longest matching route wins, and misses are recorded.

    Routes are matched by substring with the longest key first, so a test can register both
    ``/repos/acme/tool`` and ``/repos/acme/tool/git/trees/main`` without depending on the
    order they were written in.
    """

    def __init__(
        self,
        routes: dict[str, Any],
        texts: dict[str, str] | None = None,
        *,
        rate_headers: dict[str, str] | None = None,
    ) -> None:
        """Record the routes, the raw texts, and the rate-limit headers to answer with."""
        self.routes = routes
        self.texts = texts or {}
        self.rate_headers = dict(RATE_HEADERS if rate_headers is None else rate_headers)
        self.requests: list[str] = []
        self.misses: list[str] = []

    def __call__(self, url: str) -> tuple[int, dict[str, str], Any]:
        """Answer one request from the recorded routes."""
        self.requests.append(url)
        for needle in sorted(self.texts, key=len, reverse=True):
            if needle in url:
                return 200, dict(self.rate_headers), self.texts[needle]
        for needle in sorted(self.routes, key=len, reverse=True):
            if needle in url:
                return 200, dict(self.rate_headers), self.routes[needle]
        self.misses.append(url)
        return 404, dict(self.rate_headers), {"message": "Not Found"}

    def urls_like(self, needle: str) -> list[str]:
        """Return the requests whose URL contains a substring."""
        return [url for url in self.requests if needle in url]


def search_route(query: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return a search route keyed the way the client encodes its query string."""
    return urllib.parse.quote(query), payload


def client_for(
    fake: FakeGithub, *, ceiling: int = 5000
) -> tuple[GithubClient, RequestBudget]:
    """Return a budgeted client reading from a recorded API."""
    budget = RequestBudget(ceiling=ceiling)
    return GithubClient(budget, fetch=fake), budget


def commit(
    sha: str,
    *,
    login: str = "someone",
    app: str = "",
    name: str = "",
    email: str = "",
    date: str = "2026-09-01T10:00:00Z",
) -> dict[str, Any]:
    """Build a commit payload shaped like GitHub's commit list entries."""
    author: dict[str, Any] = {"type": "User"}
    if login:
        author = {
            "login": login,
            "type": "Bot" if login.endswith("[bot]") or login == "Copilot" else "User",
            "html_url": f"https://github.com/apps/{app}" if app else f"https://github.com/{login}",
        }
    return {
        "sha": sha,
        "author": author,
        "commit": {
            "author": {"name": name or login or "Human", "email": email, "date": date},
            "committer": {"name": name or login or "Human", "email": email, "date": date},
        },
    }


def copilot_commit(sha: str, *, date: str = "2026-09-01T10:00:00Z") -> dict[str, Any]:
    """Build a Copilot commit exactly as the live API reported one."""
    return commit(
        sha,
        login="Copilot",
        app="copilot-swe-agent",
        name="copilot-swe-agent[bot]",
        email="198982749+Copilot@users.noreply.github.com",
        date=date,
    )


def two_repository_fake() -> FakeGithub:
    """Return a recorded API holding two agent-active repositories."""
    query = "is:pr is:merged author:app/copilot-swe-agent merged:2026-09-01..2026-09-30"
    routes: dict[str, Any] = {}
    texts: dict[str, str] = {}
    items: list[dict[str, Any]] = []
    for owner, repo in (("acme", "tool"), ("acme", "other")):
        prefix = f"/repos/{owner}/{repo}"
        routes[f"{prefix}/pulls/7/files?per_page=100"] = []
        routes[f"{prefix}/pulls/7"] = {
            "number": 7,
            "html_url": f"https://github.com/{owner}/{repo}/pull/7",
            "merged_at": "2026-09-02T12:00:00Z",
            "merge_commit_sha": "merge" + repo[0] * 35,
            "user": {"login": "Copilot"},
        }
        routes[f"{prefix}/git/trees/main?recursive=1"] = {
            "truncated": False,
            "tree": [{"path": "src/agent.py", "type": "blob", "size": 400}],
        }
        routes[f"{prefix}/commits?path=src/agent.py"] = [copilot_commit(f"{repo}-sha")]
        routes[f"{prefix}/commits/{repo}-sha"] = {
            "files": [{"filename": "src/agent.py", "additions": 5}]
        }
        routes[prefix] = {"default_branch": "main", "pushed_at": "2026-09-03T00:00:00Z"}
        texts[f"raw.githubusercontent.com/{owner}/{repo}/main/src/agent.py"] = AGENT_FILE_LINES
        items.append({
            "html_url": f"https://github.com/{owner}/{repo}/pull/7",
            "repository_url": f"https://api.github.com/repos/{owner}/{repo}",
            "number": 7,
            "user": {"login": "Copilot"},
            "pull_request": {"merged_at": "2026-09-02T12:00:00Z"},
        })
    routes.update([search_route(query, {"total_count": 2, "items": items})])
    return FakeGithub(routes, texts)


def repository_routes(
    *,
    owner: str = "acme",
    repo: str = "tool",
    number: int = 7,
    tree: dict[str, Any] | None = None,
    histories: dict[str, Any] | None = None,
    texts: dict[str, str] | None = None,
    pull: dict[str, Any] | None = None,
    changed: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build one repository's routes and raw texts."""
    prefix = f"/repos/{owner}/{repo}"
    routes: dict[str, Any] = {
        f"{prefix}/pulls/{number}/files?per_page=100": [
            {"filename": path, "additions": 10, "status": "modified"}
            for path in (changed or [])
        ],
        f"{prefix}/pulls/{number}": pull
        or {
            "number": number,
            "html_url": f"https://github.com/{owner}/{repo}/pull/{number}",
            "merged_at": "2026-09-02T12:00:00Z",
            "merge_commit_sha": "merge" + "0" * 35,
            "user": {"login": "Copilot"},
        },
        f"{prefix}/git/trees/main?recursive=1": tree
        or {
            "truncated": False,
            "tree": [
                {"path": "src/agent.py", "type": "blob", "size": 400},
                {"path": "src/human.py", "type": "blob", "size": 300},
                {"path": "node_modules/vendored.js", "type": "blob", "size": 900},
            ],
        },
        prefix: {"default_branch": "main", "pushed_at": "2026-09-03T00:00:00Z"},
    }
    for path, history in (histories or {}).items():
        quoted = urllib.parse.quote(path)
        routes[f"{prefix}/commits?path={quoted}"] = history
    raw = {f"raw.githubusercontent.com/{owner}/{repo}/main/{path}": text for path, text in (texts or {}).items()}
    return routes, raw


AGENT_FILE_LINES = "\n".join(f"line {index}" for index in range(200)) + "\n"
HUMAN_FILE_LINES = "\n".join(f"line {index}" for index in range(120)) + "\n"


class TestSearchPartitioning(unittest.TestCase):
    """Check the search side: real partitions, honoured caps, and refusals."""

    def test_months_are_partitioned_and_clipped_to_the_range(self) -> None:
        """Cover an inclusive range with calendar months, clipping the first and last."""
        self.assertEqual(
            month_windows("2026-01-15", "2026-03-04"),
            [("2026-01-15", "2026-01-31"), ("2026-02-01", "2026-02-28"),
             ("2026-03-01", "2026-03-04")],
        )

    def test_a_backwards_range_is_refused(self) -> None:
        """Refuse a window that ends before it starts."""
        with self.assertRaises(SystemExit):
            month_windows("2026-03-01", "2026-02-01")

    def test_halving_splits_a_window_without_overlap(self) -> None:
        """Split a window into two adjacent halves covering the same days."""
        self.assertEqual(halve(("2026-09-01", "2026-09-30")),
                         [("2026-09-01", "2026-09-15"), ("2026-09-16", "2026-09-30")])

    def test_a_window_over_the_result_cap_is_split_and_recorded(self) -> None:
        """Record the query that revealed the cap, then the narrower queries that beat it."""
        month = "is:pr is:merged author:app/copilot-swe-agent merged:2026-09-01..2026-09-30"
        early = "is:pr is:merged author:app/copilot-swe-agent merged:2026-09-01..2026-09-15"
        fake = FakeGithub(dict([
            search_route(month, {"total_count": 1500, "items": []}),
            search_route(early, {
                "total_count": 3,
                "items": [
                    {"html_url": "https://github.com/acme/one/pull/1",
                     "repository_url": "https://api.github.com/repos/acme/one",
                     "number": 1, "user": {"login": "Copilot"},
                     "pull_request": {"merged_at": "2026-09-10T00:00:00Z"}},
                    {"html_url": "https://github.com/acme/two/pull/2",
                     "repository_url": "https://api.github.com/repos/acme/two",
                     "number": 2, "user": {"login": "Copilot"},
                     "pull_request": {"merged_at": "2026-09-11T00:00:00Z"}},
                ],
            }),
        ]))
        client, _ = client_for(fake)
        candidates, partitions = search_merged(
            client, agent_for_slug("copilot-swe-agent"), since="2026-09-01",
            until="2026-09-30", limit=10,
        )

        self.assertEqual([item.repository for item in candidates],
                         ["https://github.com/acme/one", "https://github.com/acme/two"])
        capped = [item for item in partitions if item.capped]
        self.assertEqual(len(capped), 1)
        self.assertEqual(capped[0].total_count, 1500)
        self.assertEqual(capped[0].query, month)
        self.assertTrue(all(not item.capped for item in partitions if item is not capped[0]))

    def test_the_per_agent_limit_is_a_ceiling_on_repositories(self) -> None:
        """Stop reading results once the requested number has been reached."""
        query = "is:pr is:merged author:app/cursor merged:2026-09-01..2026-09-30"
        items = [
            {"html_url": f"https://github.com/acme/r{index}/pull/1",
             "repository_url": f"https://api.github.com/repos/acme/r{index}",
             "number": 1, "user": {"login": "cursor[bot]"},
             "pull_request": {"merged_at": "2026-09-05T00:00:00Z"}}
            for index in range(5)
        ]
        fake = FakeGithub(dict([search_route(query, {"total_count": 5, "items": items})]))
        client, _ = client_for(fake)
        candidates, _ = search_merged(
            client, agent_for_slug("cursor"), since="2026-09-01", until="2026-09-30", limit=2
        )

        self.assertEqual(len(candidates), 2)

    def test_an_unregistered_app_slug_is_reported_rather_than_swallowed(self) -> None:
        """Raise on the 422 an unknown app slug produces instead of reading it as no work."""
        class Rejecting(FakeGithub):
            def __call__(self, url: str) -> tuple[int, dict[str, str], Any]:
                self.requests.append(url)
                return 422, dict(self.rate_headers), {"message": "Validation Failed"}

        client, _ = client_for(Rejecting({}))
        with self.assertRaises(GithubError):
            search_merged(client, agent_for_slug("devin-ai-integration"),
                          since="2026-09-01", until="2026-09-30", limit=5)


class TestAgentRegistry(unittest.TestCase):
    """Check that agent authorship is recognised, and that a bot is not AI by default."""

    def test_the_app_page_names_the_agent(self) -> None:
        """Identify an agent by the app page GitHub attaches to its account."""
        self.assertIsNotNone(agent_of({"login": "Copilot", "type": "Bot",
                                       "html_url": "https://github.com/apps/copilot-swe-agent"}))

    def test_the_registered_login_is_enough(self) -> None:
        """Identify an agent whose app page URL is absent but whose login is registered."""
        self.assertIsNotNone(agent_of({"login": "devin-ai-integration[bot]", "type": "Bot"}))

    def test_an_unattributed_commit_is_identified_by_its_git_author(self) -> None:
        """Fall back to the git author when GitHub did not attribute the commit to an app."""
        found = agent_of(None, {"name": "copilot-swe-agent[bot]",
                                "email": "198982749+Copilot@users.noreply.github.com"})

        self.assertIsNotNone(found)
        self.assertEqual(found.slug, "copilot-swe-agent")

    def test_an_unregistered_bot_is_not_ai_authorship(self) -> None:
        """Leave a maintenance bot out of the registry rather than calling every bot an agent."""
        self.assertIsNone(agent_of({"login": "dependabot[bot]", "type": "Bot",
                                    "html_url": "https://github.com/apps/dependabot"}))

    def test_a_recorded_commit_identity_identifies_the_agent(self) -> None:
        """Recognise Cursor's commits, which are authored by ``cursoragent``, not ``cursor[bot]``.

        This is the measured mismatch that made the rule call agent-written files human: the
        pull request belongs to the app, but the commits inside it belong to a plain account.
        """
        by_login = agent_of({"login": "cursoragent", "type": "User",
                             "html_url": "https://github.com/cursoragent"})
        by_git_author = agent_of(None, {"name": "Cursor Agent",
                                        "email": "cursoragent@cursor.com"})

        self.assertEqual(by_login.slug, "cursor")
        self.assertEqual(by_git_author.slug, "cursor")

    def test_a_noreply_commit_address_identifies_its_agent(self) -> None:
        """Match the address GitHub attributes an app's commit to."""
        found = agent_of(None, {
            "name": "google-labs-jules[bot]",
            "email": "161369871+google-labs-jules[bot]@users.noreply.github.com",
        })

        self.assertEqual(found.slug, "google-labs-jules")

    def test_devin_commits_carry_no_agent_identity(self) -> None:
        """Record that no commit of Devin's identifies it, so only its pull request can."""
        self.assertFalse(agent_for_slug("devin-ai-integration").commit_attributable)
        self.assertIsNone(agent_of(None, {"name": "Giorgio Sandvliet",
                                         "email": "g.sandvliet88@gmail.com"}))

    def test_every_registered_slug_resolves(self) -> None:
        """Keep the registry's slugs and logins able to be looked up."""
        for slug in AGENT_SLUGS:
            self.assertIsNotNone(agent_for_slug(slug))


class TestAttribution(unittest.TestCase):
    """Check the last-touch rule and the bound on an attributed line count."""

    def test_the_newest_agent_touch_attributes_the_file(self) -> None:
        """Attribute a file whose most recent commit was an agent's."""
        attribution, agent = classify_touches([copilot_commit("a"), commit("b")])

        self.assertEqual(attribution, AGENT_FILE)
        self.assertEqual(agent.slug, "copilot-swe-agent")

    def test_a_file_with_no_agent_touch_is_the_human_class(self) -> None:
        """Attribute a file to the human class when no examined touch was an agent's."""
        attribution, agent = classify_touches([commit("a"), commit("b"), commit("c")])

        self.assertEqual(attribution, HUMAN_FILE)
        self.assertIsNone(agent)

    def test_an_older_agent_touch_is_undecided_rather_than_human(self) -> None:
        """Leave a file an agent changed earlier out of both classes."""
        attribution, _ = classify_touches(
            [commit("a"), commit("b"), copilot_commit("c"), commit("d")]
        )

        self.assertEqual(attribution, UNKNOWN_FILE)

    def test_an_empty_history_is_undecided(self) -> None:
        """Refuse to attribute a file whose history could not be read."""
        self.assertEqual(classify_touches([])[0], UNKNOWN_FILE)

    def test_the_attributed_count_is_bounded_by_the_current_length(self) -> None:
        """Cap an attributed count by the file's present length, not the commit's additions."""
        truth = file_truth(
            "src/agent.py", [copilot_commit("a")], additions=40, current_lines=12
        )

        self.assertEqual(truth.ai_lines, 12)

    def test_a_human_file_carries_no_attributed_lines(self) -> None:
        """Attribute no AI lines to a file no agent last touched."""
        truth = file_truth("src/human.py", [commit("a")], additions=0, current_lines=90)

        self.assertEqual(truth.ai_lines, 0)
        self.assertEqual(truth.current_lines, 90)

    def test_the_window_bounds_the_human_claim(self) -> None:
        """Decide the human class over the examined window, and record how many were examined."""
        history = [commit(f"c{index}") for index in range(6)]
        truth = file_truth("src/human.py", history, window=3, current_lines=10)

        self.assertEqual(truth.attribution, HUMAN_FILE)
        self.assertEqual(truth.touches_examined, 3)


class TestHarvestRepository(unittest.TestCase):
    """Check the per-repository read: what it records, and what it refuses to spend on."""

    def _routes(self, **overrides):
        histories = {
            "src/agent.py": [copilot_commit("agent-sha")],
            "src/human.py": [commit("human-sha"), commit("older-sha")],
        }
        return repository_routes(
            histories=histories,
            texts={"src/agent.py": AGENT_FILE_LINES, "src/human.py": HUMAN_FILE_LINES},
            **overrides,
        )

    def test_a_repository_records_both_classes_with_their_commits(self) -> None:
        """Record the agent file, the human file, and the commit each attribution rests on."""
        routes, texts = self._routes()
        routes["/repos/acme/tool/commits/agent-sha"] = {
            "files": [{"filename": "src/agent.py", "additions": 150}]
        }
        fake = FakeGithub(routes, texts)
        client, _ = client_for(fake)
        candidate = MergeCandidate(
            repository="https://github.com/acme/tool",
            repository_api="https://api.github.com/repos/acme/tool",
            number=7, agent="copilot-swe-agent", author="Copilot",
            merged_at="2026-09-02T12:00:00Z",
        )
        truth = harvest_repository(client, candidate, max_files=10)

        self.assertEqual(truth.rejected, "")
        self.assertEqual([item.path for item in truth.files_with(AGENT_FILE)], ["src/agent.py"])
        self.assertEqual([item.path for item in truth.files_with(HUMAN_FILE)], ["src/human.py"])
        agent_file = truth.files_with(AGENT_FILE)[0]
        self.assertEqual(agent_file.last_touch_sha, "agent-sha")
        self.assertEqual(agent_file.last_touch_agent, "copilot-swe-agent")
        self.assertEqual(agent_file.ai_lines, 150)
        self.assertEqual(truth.lines_with(HUMAN_FILE), 120)
        self.assertEqual(truth.pull_request.merge_commit, "merge" + "0" * 35)
        self.assertAlmostEqual(truth.attributed_fraction or 0, 150 / 270, places=6)

    def test_files_outside_the_selection_rules_are_never_requested(self) -> None:
        """Skip a path the reader would exclude, so it cannot decide a label or cost a request."""
        routes, texts = self._routes()
        routes["/repos/acme/tool/commits/agent-sha"] = {
            "files": [{"filename": "src/agent.py", "additions": 150}]
        }
        fake = FakeGithub(routes, texts)
        client, _ = client_for(fake)
        harvest_repository(
            client,
            MergeCandidate("https://github.com/acme/tool",
                           "https://api.github.com/repos/acme/tool", 7,
                           "copilot-swe-agent", "Copilot", "2026-09-02T12:00:00Z"),
            max_files=10,
        )

        self.assertEqual(fake.urls_like("vendored"), [])

    def test_the_file_cap_bounds_what_is_read(self) -> None:
        """Examine no more files than the cap allows."""
        tree = {
            "truncated": False,
            "tree": [
                {"path": f"src/module_{index}.py", "type": "blob", "size": 100}
                for index in range(6)
            ],
        }
        histories = {f"src/module_{index}.py": [commit(f"c{index}")] for index in range(6)}
        routes, texts = repository_routes(tree=tree, histories=histories, texts={})
        fake = FakeGithub(routes, texts)
        client, _ = client_for(fake)
        truth = harvest_repository(
            client,
            MergeCandidate("https://github.com/acme/tool",
                           "https://api.github.com/repos/acme/tool", 7,
                           "cursor", "cursor[bot]", "2026-09-02T12:00:00Z"),
            max_files=2,
        )

        self.assertEqual(len(truth.files), 2)
        self.assertEqual(len(fake.urls_like("/commits?path=")), 2)

    def test_an_unreadable_repository_is_rejected_rather_than_fatal(self) -> None:
        """Record a repository whose tree cannot be read instead of raising."""
        fake = FakeGithub({"/pulls/7": {"number": 7, "merge_commit_sha": "x" * 40}})
        client, _ = client_for(fake)
        truth = harvest_repository(
            client,
            MergeCandidate("https://github.com/acme/gone",
                           "https://api.github.com/repos/acme/gone", 7,
                           "cursor", "cursor[bot]", ""),
        )

        self.assertEqual(truth.rejected, "repository metadata could not be read")

    def test_the_pull_requests_own_paths_are_examined_before_the_ranking(self) -> None:
        """Examine what the agent changed first, whatever the reader's ranking prefers.

        The live sample behind this rule changed a log document and three SQL migrations while
        the top-ranked paths were the project's own TypeScript modules, so a read that followed
        only the ranking examined five files the agent had never touched.
        """
        tree = {
            "tree": [
                {"path": "web/src/index.ts", "type": "blob", "size": 200},
                {"path": "web/docs/logs/check.md", "type": "blob", "size": 200},
            ]
        }

        self.assertEqual(
            [path for path, _ in candidate_files(
                tree, max_files=1, changed=["web/docs/logs/check.md"])],
            ["web/docs/logs/check.md"],
        )

    def test_a_cursor_pull_request_attributes_its_own_committed_file(self) -> None:
        """Read the file the pull request changed and attribute it to the commit identity."""
        tree = {
            "truncated": False,
            "tree": [
                {"path": "web/src/index.ts", "type": "blob", "size": 200},
                {"path": "web/docs/logs/check.md", "type": "blob", "size": 200},
            ],
        }
        histories = {
            "web/src/index.ts": [commit("human-sha", login="maintainer",
                                        date="2026-08-05T00:00:00Z")],
            "web/docs/logs/check.md": [
                commit("agent-sha", login="cursoragent", name="Cursor Agent",
                       email="cursoragent@cursor.com", date="2026-10-13T00:00:00Z")
            ],
        }
        routes, texts = repository_routes(
            number=9,
            tree=tree,
            histories=histories,
            texts={"web/src/index.ts": HUMAN_FILE_LINES,
                   "web/docs/logs/check.md": AGENT_FILE_LINES},
            pull={"number": 9,
                  "html_url": "https://github.com/acme/tool/pull/9",
                  "merged_at": "2026-10-14T12:00:00Z",
                  "merge_commit_sha": "m" * 40,
                  "user": {"login": "cursor[bot]"}},
            changed=["web/docs/logs/check.md"],
        )
        routes["/repos/acme/tool/commits/agent-sha"] = {
            "files": [{"filename": "web/docs/logs/check.md", "additions": 62}]
        }
        fake = FakeGithub(routes, texts)
        client, _ = client_for(fake)
        truth = harvest_repository(
            client,
            MergeCandidate("https://github.com/acme/tool",
                           "https://api.github.com/repos/acme/tool", 9,
                           "cursor", "cursor[bot]", "2026-10-14T12:00:00Z"),
            max_files=1,
        )

        self.assertEqual([item.path for item in truth.files], ["web/docs/logs/check.md"])
        self.assertEqual(truth.files[0].attribution, AGENT_FILE)
        self.assertEqual(truth.files[0].last_touch_agent, "cursor")
        self.assertEqual(truth.files[0].ai_lines, 62)
        self.assertEqual(truth.pull_request.merge_commit, "m" * 40)
        self.assertEqual(truth.lines_with(AGENT_FILE), 62)

    def test_the_candidate_file_order_matches_the_reader(self) -> None:
        """Rank source before metadata and documentation, as the evidence view does."""
        tree = {
            "tree": [
                {"path": "README.md", "type": "blob", "size": 200},
                {"path": "package.json", "type": "blob", "size": 200},
                {"path": "src/app.py", "type": "blob", "size": 200},
                {"path": "assets/logo.png", "type": "blob", "size": 200},
            ]
        }

        self.assertEqual(
            [path for path, _ in candidate_files(tree, max_files=10)],
            ["src/app.py", "package.json", "README.md"],
        )


class TestHarvestLoop(unittest.TestCase):
    """Check resumption, the budget stop, and what the cohort file ends up holding."""

    def _fake(self) -> FakeGithub:
        query = "is:pr is:merged author:app/copilot-swe-agent merged:2026-09-01..2026-09-30"
        routes, texts = repository_routes(
            histories={
                "src/agent.py": [copilot_commit("agent-sha")],
                "src/human.py": [commit("human-sha")],
            },
            texts={"src/agent.py": AGENT_FILE_LINES, "src/human.py": HUMAN_FILE_LINES},
        )
        routes["/repos/acme/tool/commits/agent-sha"] = {
            "files": [{"filename": "src/agent.py", "additions": 150}]
        }
        routes.update([search_route(query, {
            "total_count": 1,
            "items": [
                {"html_url": "https://github.com/acme/tool/pull/7",
                 "repository_url": "https://api.github.com/repos/acme/tool",
                 "number": 7, "user": {"login": "Copilot"},
                 "pull_request": {"merged_at": "2026-09-02T12:00:00Z"}}
            ],
        })])
        return FakeGithub(routes, texts)

    def _harvest(self, path: str, fake: FakeGithub, *, ceiling: int = 5000):
        store = CohortStore(path)
        client, budget = client_for(fake, ceiling=ceiling)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            summary = harvest(client, since="2026-09-01", until="2026-09-30",
                              per_agent_limit=5, max_repositories=10, store=store)
        return summary, budget, captured.getvalue()

    def test_a_harvested_repository_lands_in_the_cohort_file(self) -> None:
        """Append the repository record so a later run reads it back."""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "cohort.jsonl")
            summary, budget, printed = self._harvest(path, self._fake())

            self.assertEqual(summary.repositories, 1)
            self.assertEqual(summary.stopped, "")
            cohort = read_cohort(path)
            self.assertEqual(len(cohort.repositories), 1)
            self.assertEqual(cohort.repositories[0].url, "https://github.com/acme/tool")
            self.assertGreater(budget.spent, 0)
            self.assertIn("agent file(s)", printed)

    def test_a_second_run_skips_what_the_cohort_already_holds(self) -> None:
        """Resume from the appended cohort instead of re-reading the same repository."""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "cohort.jsonl")
            self._harvest(path, self._fake())
            second = self._fake()
            summary, _, _ = self._harvest(path, second)

            self.assertEqual(summary.repositories, 1)
            self.assertEqual(second.urls_like("/git/trees/"), [])
            self.assertEqual(second.urls_like("/pulls/"), [])
            self.assertEqual(len(read_cohort(path).repositories), 1)

    def test_refresh_re_reads_a_recorded_repository(self) -> None:
        """Re-harvest when asked, so a changed rule version can be applied."""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "cohort.jsonl")
            self._harvest(path, self._fake())
            refreshed = self._fake()
            store = CohortStore(path)
            client, _ = client_for(refreshed)
            with contextlib.redirect_stdout(io.StringIO()):
                harvest(client, since="2026-09-01", until="2026-09-30", per_agent_limit=5,
                        max_repositories=10, store=store, refresh=True)

            self.assertEqual(len(refreshed.urls_like("/git/trees/")), 1)

    def test_the_budget_stops_the_run_and_keeps_the_finished_cohort(self) -> None:
        """Stop cleanly at the ceiling instead of starting a repository it cannot finish."""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "cohort.jsonl")
            summary, budget, _ = self._harvest(path, self._fake(), ceiling=2)

            self.assertEqual(summary.repositories, 0)
            self.assertIn("ceiling", summary.stopped)
            self.assertEqual(budget.ceiling_hits, 0)
            self.assertEqual(read_cohort(path).repositories, [])

    def test_a_ceiling_hit_mid_repository_keeps_the_finished_records(self) -> None:
        """Stop inside the second repository and still hold the first one complete."""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "cohort.jsonl")
            store = CohortStore(path)
            # Ranking reads both candidates' pull requests, repositories, and trees before the
            # first repository is harvested, and the first repository reuses three of those
            # payloads. A ceiling of 11 to 13 completes the first repository and stops inside the
            # second; 12 is the middle of that range.
            client, budget = client_for(two_repository_fake(), ceiling=12)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = harvest(client, since="2026-09-01", until="2026-09-30",
                                  per_agent_limit=5, max_repositories=10, store=store,
                                  max_files=1)

            self.assertIn("ceiling", summary.stopped)
            self.assertEqual(summary.repositories, 1)
            self.assertLessEqual(budget.spent, budget.ceiling)
            cohort = read_cohort(path)
            self.assertEqual([item.url for item in cohort.repositories],
                             ["https://github.com/acme/tool"])
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        self.assertIn(json.loads(line)["kind"], {"repository", "human"})

    def test_the_budget_refuses_the_call_that_would_pass_the_ceiling(self) -> None:
        """Raise rather than spend past the ceiling."""
        budget = RequestBudget(ceiling=1)
        budget.spend()
        with self.assertRaises(BudgetExceeded):
            budget.spend()

    def test_the_observed_rate_limit_is_recorded(self) -> None:
        """Keep the limit GitHub reported, which is what sizes the next run."""
        budget = RequestBudget(ceiling=10)
        budget.observe({"x-ratelimit-limit": "10", "x-ratelimit-remaining": "7"})

        self.assertEqual(budget.to_dict()["rate_limit"], 10)
        self.assertEqual(budget.to_dict()["rate_remaining"], 7)


class TestHumanClass(unittest.TestCase):
    """Check the recorded no-agent protocol and its rejections."""

    def _routes(self, *, pushed: str, commits: list[dict[str, Any]]):
        return {
            "/repos/old/project/commits": commits,
            "/repos/old/project": {"default_branch": "main", "pushed_at": pushed},
        }

    def test_an_old_repository_with_no_agent_commits_is_verified(self) -> None:
        """Verify the class and record the protocol that ruled it in."""
        commits = [commit(f"sha{index}", date=f"2018-0{index + 1}-01T00:00:00Z")
                   for index in range(4)]
        fake = FakeGithub(self._routes(pushed="2019-05-01T00:00:00Z", commits=commits))
        client, _ = client_for(fake)
        verification, reason = verify_human_repository(
            client, "https://github.com/old/project", cutoff="2023-01-01"
        )

        self.assertEqual(reason, "")
        self.assertIsNotNone(verification)
        self.assertEqual(verification.commits_examined, 4)
        self.assertIn("predates 2023-01-01", verification.protocol)
        self.assertIn("no registered agent", verification.protocol)

    def test_a_repository_pushed_after_the_cutoff_is_rejected(self) -> None:
        """Reject a repository whose last push is not before the cutoff."""
        fake = FakeGithub(self._routes(pushed="2026-01-01T00:00:00Z", commits=[]))
        client, _ = client_for(fake)
        verification, reason = verify_human_repository(
            client, "https://github.com/old/project", cutoff="2023-01-01"
        )

        self.assertIsNone(verification)
        self.assertIn("not before 2023-01-01", reason)

    def test_an_agent_commit_rejects_the_whole_repository(self) -> None:
        """Reject a repository an agent has touched, whoever touched it last."""
        commits = [commit("a", date="2018-01-01T00:00:00Z"), copilot_commit("b", date="2018-02-01T00:00:00Z")]
        fake = FakeGithub(self._routes(pushed="2019-05-01T00:00:00Z", commits=commits))
        client, _ = client_for(fake)
        verification, reason = verify_human_repository(
            client, "https://github.com/old/project", cutoff="2023-01-01"
        )

        self.assertIsNone(verification)
        self.assertIn("copilot-swe-agent", reason)

    def test_a_recent_commit_rejects_the_repository(self) -> None:
        """Reject when any inspected commit is newer than the cutoff."""
        commits = [commit("a", date="2024-01-01T00:00:00Z")]
        fake = FakeGithub(self._routes(pushed="2019-05-01T00:00:00Z", commits=commits))
        client, _ = client_for(fake)
        verification, reason = verify_human_repository(
            client, "https://github.com/old/project", cutoff="2023-01-01"
        )

        self.assertIsNone(verification)
        self.assertIn("newest commit", reason)

    def test_an_unreadable_history_is_rejected(self) -> None:
        """Reject a repository whose commit history cannot be read."""
        fake = FakeGithub({"/repos/old/project": {"pushed_at": "2019-01-01T00:00:00Z"}})
        client, _ = client_for(fake)
        verification, reason = verify_human_repository(
            client, "https://github.com/old/project", cutoff="2023-01-01"
        )

        self.assertIsNone(verification)
        self.assertEqual(reason, "commit history could not be read")


class TestLabelManifest(unittest.TestCase):
    """Check that the cohort's labels meet the same standard the data builder applies."""

    def _cohort(self):
        """Harvest one repository into a temporary cohort and read it back."""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "cohort.jsonl")
            client, _ = client_for(TestHarvestLoop()._fake())
            with contextlib.redirect_stdout(io.StringIO()):
                harvest(client, since="2026-09-01", until="2026-09-30",
                        per_agent_limit=5, max_repositories=10, store=CohortStore(path))
            return read_cohort(path)

    def test_the_ai_entry_carries_the_pull_request_and_its_merge_commit(self) -> None:
        """Build an entry the fail-closed standard accepts, resting on the merge commit."""
        cohort = self._cohort()
        entries, dropped = manifest.derive_entries(
            cohort, audited_by="tester", audited_at="2026-10-15"
        )
        accepted, rejected = validate_entries(entries)

        self.assertEqual(dropped, [])
        self.assertEqual(rejected, [], [item.reason for item in rejected])
        self.assertEqual(len(accepted), 1)
        entry = entries[0]
        self.assertEqual(entry["label"], "ai")
        self.assertEqual(entry["evidence"][0]["agent"], "copilot-swe-agent")
        self.assertTrue(entry["evidence"][0]["pull_request"].endswith("/pull/7"))
        self.assertEqual(entry["evidence"][0]["commit"], "merge" + "0" * 35)
        self.assertEqual(entry["attribution_rule"], "last-touch-v1")

    def test_a_repository_without_a_merge_commit_is_left_out_with_a_reason(self) -> None:
        """Drop an unlabelable repository instead of writing an entry that would be refused."""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "cohort.jsonl")
            routes, texts = repository_routes(
                histories={"src/agent.py": [copilot_commit("agent-sha")]},
                texts={"src/agent.py": AGENT_FILE_LINES},
                pull={"number": 7, "merged_at": "2026-09-02T12:00:00Z",
                      "merge_commit_sha": "", "user": {"login": "Copilot"}},
            )
            routes["/repos/acme/tool/commits/agent-sha"] = {
                "files": [{"filename": "src/agent.py", "additions": 10}]
            }
            query = "is:pr is:merged author:app/copilot-swe-agent merged:2026-09-01..2026-09-30"
            routes.update([search_route(query, {
                "total_count": 1,
                "items": [{"html_url": "https://github.com/acme/tool/pull/7",
                           "repository_url": "https://api.github.com/repos/acme/tool",
                           "number": 7, "user": {"login": "Copilot"},
                           "pull_request": {"merged_at": "2026-09-02T12:00:00Z"}}],
            })])
            client, _ = client_for(FakeGithub(routes, texts))
            with contextlib.redirect_stdout(io.StringIO()):
                harvest(client, since="2026-09-01", until="2026-09-30", per_agent_limit=5,
                        max_repositories=10, store=CohortStore(path))
            entries, dropped = manifest.derive_entries(
                read_cohort(path), audited_by="tester", audited_at="2026-10-15"
            )

        self.assertEqual(entries, [])
        self.assertEqual(len(dropped), 1)
        self.assertIn("merge commit", dropped[0][1])

    def test_a_manifest_with_one_class_is_refused_by_the_builder_gate(self) -> None:
        """Refuse a cohort that covers one class, which is what require_usable is for."""
        cohort = self._cohort()
        entries, _ = manifest.derive_entries(
            cohort, audited_by="tester", audited_at="2026-10-15"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "labels.jsonl")
            manifest.write_manifest(path, entries)
            loaded = load_repository_labels(path)

            self.assertEqual(loaded.counts(), {"ai": 1, "human": 0})
            with self.assertRaises(SystemExit):
                require_usable(loaded)

    def test_both_classes_together_are_usable(self) -> None:
        """Accept a manifest holding a verified human repository beside the AI entries."""
        cohort = self._cohort()
        entries, _ = manifest.derive_entries(
            cohort, audited_by="tester", audited_at="2026-10-15"
        )
        entries.append(
            manifest.human_entry(
                HumanVerification(url="https://github.com/old/project",
                                  protocol="checked 2018 commits, no agent app",
                                  commits_examined=40, newest_commit="2018-06-01",
                                  branch="main"),
                audited_by="tester", audited_at="2026-10-15",
            )
        )
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "labels.jsonl")
            manifest.write_manifest(path, entries)
            loaded = load_repository_labels(path)

            require_usable(loaded)
            self.assertEqual(loaded.counts(), {"ai": 1, "human": 1})
            self.assertEqual(loaded.label_for("https://github.com/acme/tool"), "ai")
            self.assertEqual(loaded.label_for("https://github.com/old/project"), "human")

    def test_a_self_declared_readme_is_still_not_evidence(self) -> None:
        """Reject a hand-written entry that only claims AI authorship."""
        accepted, rejected = validate_entries([
            {"url": "https://github.com/acme/claimed", "label": "ai",
             "evidence": [{"evidence_class": "self_declared_readme"}],
             "audited_by": "tester", "audited_at": "2026-10-15"}
        ])

        self.assertEqual(accepted, {})
        self.assertEqual(len(rejected), 1)


class TestRankCandidates(unittest.TestCase):
    """Check that the repositories an agent changed most are harvested first."""

    @staticmethod
    def _candidate(repo: str) -> MergeCandidate:
        """Build one merged pull request candidate in the acme organisation."""
        return MergeCandidate(
            repository=f"https://github.com/acme/{repo}",
            repository_api=f"https://api.github.com/repos/acme/{repo}",
            number=1,
            agent="copilot-swe-agent",
            author="Copilot",
            merged_at="2026-09-02T12:00:00Z",
        )

    @staticmethod
    def _fake() -> FakeGithub:
        """Record a repository where the agent changed 6 of 8 files and one where it changed 1 of 4."""
        routes: dict[str, Any] = {}
        for repo, changed, blobs in (("big", 6, 8), ("small", 1, 4)):
            prefix = f"/repos/acme/{repo}"
            routes[f"{prefix}/pulls/1"] = {"changed_files": changed}
            routes[prefix] = {"default_branch": "main"}
            routes[f"{prefix}/git/trees/main"] = {
                "tree": [{"path": f"f{index}.py", "type": "blob", "size": 10} for index in range(blobs)]
            }
        return FakeGithub(routes)

    def test_the_repository_the_agent_changed_most_comes_first(self) -> None:
        """Order candidates by the fraction of files the pull request changed, not by search order."""
        client, _ = client_for(self._fake())
        ranked = rank_candidates(
            client, [self._candidate("small"), self._candidate("big")], known=set(), store={}
        )

        self.assertEqual([c.repository for c in ranked], [
            "https://github.com/acme/big", "https://github.com/acme/small",
        ])

    def test_a_repository_already_in_the_cohort_is_not_ranked_or_read(self) -> None:
        """Drop known repositories before spending any request on them."""
        fake = self._fake()
        client, _ = client_for(fake)
        ranked = rank_candidates(
            client,
            [self._candidate("small"), self._candidate("big")],
            known={"https://github.com/acme/big"},
            store={},
        )

        self.assertEqual([c.repository for c in ranked], ["https://github.com/acme/small"])
        self.assertEqual(fake.urls_like("/repos/acme/big"), [])


class TestHarvestCommand(unittest.TestCase):
    """Check the command's own behaviour: refusals and a run that writes its artifacts."""

    def test_an_unregistered_agent_is_refused_before_any_request(self) -> None:
        """Refuse a slug that would return HTTP 422 from GitHub."""
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(["--agents", "devin", "--dry-run"])

    def test_a_dry_run_spends_nothing_and_prints_the_plan(self) -> None:
        """Plan the partitions without a request, from the date range alone."""
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "cohort.jsonl")
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                code = cli.main([
                    "--agents", "cursor", "--since", "2026-09-01", "--until", "2026-10-05",
                    "--out", path, "--dry-run",
                ])

            printed = captured.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("author:app/cursor", printed)
            self.assertIn("2026-09-01..2026-09-30", printed)
            self.assertFalse(os.path.exists(path))

    def test_a_run_writes_the_cohort_provenance_and_manifest(self) -> None:
        """Write the cohort, its provenance, and a manifest the label standard accepts."""
        with tempfile.TemporaryDirectory() as folder:
            cohort_path = os.path.join(folder, "cohort.jsonl")
            labels_path = os.path.join(folder, "labels.jsonl")
            fake = TestHarvestLoop()._fake()

            def client_from_budget(budget, **_kwargs):
                return GithubClient(budget, fetch=fake)

            with mock.patch.object(cli, "GithubClient", client_from_budget), \
                    mock.patch.object(cli, "github_token", lambda: "token"), \
                    contextlib.redirect_stdout(io.StringIO()) as captured:
                code = cli.main([
                    "--agents", "copilot-swe-agent",
                    "--since", "2026-09-01", "--until", "2026-09-30",
                    "--per-agent-limit", "5", "--out", cohort_path,
                    "--labels-out", labels_path, "--audited-by", "tester",
                ])

            printed = captured.getvalue()
            self.assertEqual(code, 0)
            self.assertIn("provenance", printed)
            provenance_path = os.path.join(folder, "cohort.provenance.json")
            self.assertTrue(os.path.exists(provenance_path))
            with open(provenance_path, encoding="utf-8") as handle:
                run = json.load(handle)["runs"][0]
            self.assertEqual(run["agents"], ["copilot-swe-agent"])
            self.assertEqual(run["budget"]["spent"] > 0, True)
            self.assertEqual(run["cohort"]["repository"], 1)
            loaded = load_repository_labels(labels_path)
            self.assertEqual(loaded.counts()["ai"], 1)
            self.assertEqual(loaded.rejected, [])

    def test_a_run_that_recorded_nothing_reports_failure(self) -> None:
        """Return a non-zero status when no repository could be recorded."""
        with tempfile.TemporaryDirectory() as folder:
            cohort_path = os.path.join(folder, "cohort.jsonl")
            fake = FakeGithub({})

            def client_from_budget(budget, **_kwargs):
                return GithubClient(budget, fetch=fake)

            with mock.patch.object(cli, "GithubClient", client_from_budget), \
                    mock.patch.object(cli, "github_token", lambda: "token"), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = cli.main([
                    "--agents", "cursor", "--since", "2026-09-01",
                    "--until", "2026-09-30", "--out", cohort_path,
                ])

            self.assertEqual(code, 1)


@pytest.mark.live
class TestLiveAgentRegistry(unittest.TestCase):
    """Confirm against GitHub that the registry still describes real apps.

    This is the check that would have caught the Cursor mismatch on its own: the registry
    matched pull-request authors only, so every Cursor commit looked human. It is a live test
    because no recording can prove an app slug is still real — the offline suite proves the
    rule, and this proves the identities the rule is given.
    """

    # Rejections the harvester can record for a repository it simply cannot read. A live
    # sample that hits one is skipped, because it says nothing about the rule; any other
    # rejection is a failure, because the code has no such path.
    KNOWN_REJECTIONS = (
        "repository URL could not be parsed",
        "repository metadata could not be read",
        "file tree could not be read",
    )

    def setUp(self) -> None:
        """Give each live test its own small budget and its own paced client.

        The budget is per test on purpose: a shared ceiling made one test's search partitions
        exhaust the next test's request, which turned a spent budget into a failing assertion.
        """
        self.budget = RequestBudget(ceiling=60)
        # Anonymous search allows a small number of requests a minute, and this class asks
        # several agents in a row, so each request waits instead of being throttled.
        self.client = GithubClient(
            self.budget, fetch=None, token=github_token(), pause=3.0, attempts=2
        )

    def test_every_registered_slug_answers_and_the_author_filter_is_applied(self) -> None:
        """Prove each registered app exists and that the author qualifier really filters.

        A registered app that has gone quiet is not a failure — ``codegen-sh`` holds 4,007
        merged pull requests all time and none in a recent month, so asserting activity would
        fail on a healthy registry. Existence is what a live check can prove: an unknown slug
        answers HTTP 422 rather than zero. What is asserted beyond that is that the filter is
        applied, which an ignored qualifier fails by returning one identical count for every
        agent. The window is read at ``max_depth=0`` because a high-volume agent's quarter
        exceeds the result cap, and one capped query answers both questions for a fifth of the
        requests a full partition plan would spend.
        """
        claimed: dict[str, int] = {}
        for agent in AGENT_APPS:
            with self.subTest(agent=agent.slug):
                candidates, partitions = search_merged(
                    self.client, agent, since="2026-06-01", until="2026-09-30", limit=1,
                    max_depth=0,
                )
                claimed[agent.slug] = sum(item.total_count for item in partitions)
                if candidates:
                    self.assertTrue(candidates[0].repository.startswith("https://github.com/"))

        print(f"\n[live] merged pull requests by agent, 2026-06-01..2026-09-30: {claimed}")
        self.assertGreater(
            max(claimed.values()), 0,
            "no registered agent claimed a single merged pull request",
        )
        self.assertGreater(
            len(set(claimed.values())), 1,
            "every agent claimed the same count, which is what an ignored author filter returns",
        )

    def test_an_unknown_app_slug_is_refused(self) -> None:
        """Refuse a slug GitHub cannot resolve instead of reporting an empty cohort."""
        unknown = AgentApp("not-a-real-agent-slug", "not-real[bot]", "Not an agent")

        with self.assertRaises(GithubError):
            search_merged(self.client, unknown, since="2026-09-01", until="2026-09-30", limit=1)

    def test_a_live_repository_records_the_identities_it_was_built_on(self) -> None:
        """Harvest one repository and check every attribution names a registered agent.

        The attribution itself is not asserted: which files a sampled pull request changed is
        the repository's business, and a test that demanded agent-attributed files would fail
        whenever the sample landed on a documentation-only pull request. What is asserted is
        the invariant — an agent attribution always names a registered slug, and the record
        always carries the pull request it came from — and the observation is printed.
        """
        agent = agent_for_slug("cursor")
        candidates, _ = search_merged(
            self.client, agent, since="2026-09-01", until="2026-09-30", limit=1, max_depth=0
        )
        if not candidates:
            self.skipTest("no merged Cursor pull request in the sampled window")
        truth = harvest_repository(self.client, candidates[0], max_files=5)

        if truth.rejected:
            self.assertIn(truth.rejected, self.KNOWN_REJECTIONS)
            self.skipTest(truth.rejected)
        self.assertIsNotNone(truth.pull_request)
        self.assertTrue(truth.pull_request.merge_commit)
        for item in truth.files:
            self.assertIn(item.last_touch_agent, ("", *AGENT_SLUGS))
        print(
            f"\n[live] {truth.url}: {len(truth.files_with(AGENT_FILE))} agent / "
            f"{len(truth.files_with(HUMAN_FILE))} human / "
            f"{len(truth.files_with(UNKNOWN_FILE))} unknown of {len(truth.files)} file(s), "
            f"{truth.requests} request(s), budget {self.budget.spent}/{self.budget.ceiling}"
        )


if __name__ == "__main__":
    unittest.main()
