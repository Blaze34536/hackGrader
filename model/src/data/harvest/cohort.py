"""Harvest the agent-attributed repository cohort one repository at a time.

Each repository costs a fixed handful of requests and a few per file, so the loop is written
to never start work it cannot finish: it stops before a repository that the remaining budget
cannot cover, and every repository that finishes is appended to the cohort file immediately.
An interrupted run therefore resumes from the last recorded repository rather than from the
first, which is the difference between a cohort that can be built over several evenings and
one that has to be built in an afternoon of quota.

Only repositories whose files the grader would actually read are attributed: the candidate
file list comes from the shared selection rules the reader uses, so a path excluded from
every evidence view cannot decide a label. The truth is still a fraction over the files
examined, not over the repository, and both the examined files and the attributed lines are
recorded so a reader can rebuild the ratio with the coverage it deserves.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from repositories.selection import is_candidate, selection_priority
from repositories.urls import parse_github_url

from .agents import AGENT_APPS, AgentApp, agent_for_slug
from .attribution import TOUCH_WINDOW, classify_touches, commit_sha, file_truth
from .budget import BudgetExceeded
from .client import API_ROOT, GithubClient, RateLimited
from .queries import MergeCandidate, search_merged
from .records import (
    AGENT_FILE,
    HUMAN_FILE,
    CohortStore,
    Partition,
    PullRequest,
    RepositoryTruth,
    read_cohort,
)

# A single file large enough to be a vendored bundle spends a request per file and tells the
# cohort nothing about authorship, so the examined set is bounded by size as well as count.
MAX_FILE_BYTES = 200_000

# The cohort floor includes three languages, so the harvest reports the spread it reached.
# Extensions are mapped rather than guessed from names: a wrong language is a claim about
# the cohort's coverage, and coverage is the one thing a small cohort can overstate.
LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".swift": "swift",
    ".dart": "dart",
    ".vue": "vue",
    ".svelte": "svelte",
    ".sh": "shell",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
}


def language_of(path: str) -> str:
    """Return the language of a file path, or ``"other"`` for a non-source file."""
    lowered = path.casefold()
    for extension, language in LANGUAGE_EXTENSIONS.items():
        if lowered.endswith(extension):
            return language
    return "other"


@dataclass
class HarvestSummary:
    """What one harvest run recorded, and why it stopped if it did."""

    repositories: int = 0
    skipped: int = 0
    stopped: str = ""
    retry_after: float | None = None
    partitions: list[Partition] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def agent_files(self, truths: Iterable[RepositoryTruth]) -> int:
        """Count the agent-attributed files across recorded repositories."""
        return sum(len(item.files_with(AGENT_FILE)) for item in truths)


def candidate_files(
    tree: dict[str, Any],
    *,
    max_files: int,
    max_bytes: int = MAX_FILE_BYTES,
    changed: Iterable[str] = (),
) -> list[tuple[str, int]]:
    """Return the files to examine: the pull request's own paths, then the reader's ranking.

    The pull request's files come first because they are where the agent's work actually is,
    and the reader's ranking alone does not find them: a live sample's agent pull request
    changed a log document and three SQL migrations while the top-ranked paths were the
    project's own ``src`` modules, so a read that followed only the ranking examined five
    files the agent had never touched and an agent-written repository looked human. The
    attribution still comes from the last-touch commit, not from the diff, so this orders the
    read rather than deciding the label.
    """
    blobs = {
        str(entry.get("path") or ""): int(entry.get("size") or 0)
        for entry in tree.get("tree", [])
        if entry.get("type") == "blob"
    }

    def usable(path: str) -> bool:
        """Return whether a path is readable and inside the size bound."""
        return bool(path) and path in blobs and blobs[path] <= max_bytes and is_candidate(path)

    ordered: list[str] = []
    for path in changed:
        if usable(path) and path not in ordered:
            ordered.append(path)
    ranked = sorted(
        (path for path in blobs if usable(path) and path not in ordered),
        key=selection_priority,
    )
    ordered.extend(ranked)
    return [(path, blobs[path]) for path in ordered[:max_files]]


def _pull_request_paths(
    client: GithubClient, owner: str, repo: str, number: int
) -> list[str]:
    """Return the paths a merged pull request changed, from its first page of files.

    One page of a hundred covers the pull requests in a cohort; the reader's ranking still
    fills the examined set afterwards, so a pull request with more files than that narrows
    which of its paths are read first rather than losing them.
    """
    payload = client.get(
        f"{API_ROOT}/repos/{owner}/{repo}/pulls/{number}/files?per_page=100"
    )
    if not isinstance(payload, list):
        return []
    return [str(entry.get("filename") or "") for entry in payload if isinstance(entry, dict)]


def _pull_request(
    client: GithubClient, candidate: MergeCandidate, store: dict[str, Any]
) -> PullRequest | None:
    """Return the merged pull request record, read from the PR endpoint for its merge commit."""
    owner, repo = parse_github_url(candidate.repository)
    if not owner or not repo:
        return None
    payload = _take(client, f"{API_ROOT}/repos/{owner}/{repo}/pulls/{candidate.number}", store)
    if not isinstance(payload, dict):
        return None
    author = payload.get("user") or {}
    return PullRequest(
        url=str(payload.get("html_url") or candidate.pull_request_url),
        number=int(payload.get("number") or candidate.number),
        agent=candidate.agent,
        author=str(author.get("login") or candidate.author),
        merged_at=str(payload.get("merged_at") or candidate.merged_at),
        merge_commit=str(payload.get("merge_commit_sha") or ""),
    )


def _additions(
    client: GithubClient, owner: str, repo: str, sha: str, path: str
) -> int:
    """Return how many lines one commit added to one file."""
    if not sha:
        return 0
    payload = client.get(f"{API_ROOT}/repos/{owner}/{repo}/commits/{sha}")
    if not isinstance(payload, dict):
        return 0
    for entry in payload.get("files", []):
        if str(entry.get("filename") or "") == path:
            return int(entry.get("additions") or 0)
    return 0


def harvest_repository(
    client: GithubClient,
    candidate: MergeCandidate,
    *,
    window: int = TOUCH_WINDOW,
    max_files: int = 40,
    max_bytes: int = MAX_FILE_BYTES,
    prefetched: dict[str, Any] | None = None,
) -> RepositoryTruth:
    """Read one repository's files and attribute each to its last touch.

    A repository that cannot be read is returned with ``rejected`` set rather than raising:
    a private, renamed, or empty repository is a fact about the cohort worth recording, and
    it must not end a harvest that has budget left for real material.
    """
    owner, repo = parse_github_url(candidate.repository)
    if not owner or not repo:
        return RepositoryTruth(
            url=candidate.repository,
            pull_request=None,
            rejected="repository URL could not be parsed",
        )
    spent_before = client.budget.spent
    raw_before = client.budget.raw_spent
    store = {} if prefetched is None else prefetched
    pull_request = _pull_request(client, candidate, store)

    meta = _take(client, f"{API_ROOT}/repos/{owner}/{repo}", store)
    if not isinstance(meta, dict):
        return RepositoryTruth(
            url=candidate.repository,
            pull_request=pull_request,
            requests=client.budget.spent - spent_before,
            raw_requests=client.budget.raw_spent - raw_before,
            rejected="repository metadata could not be read",
        )
    branch = str(meta.get("default_branch") or "main")
    tree = _take(client, f"{API_ROOT}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1", store)
    if not isinstance(tree, dict):
        return RepositoryTruth(
            url=candidate.repository,
            pull_request=pull_request,
            branch=branch,
            requests=client.budget.spent - spent_before,
            raw_requests=client.budget.raw_spent - raw_before,
            rejected="file tree could not be read",
        )

    changed = _pull_request_paths(client, owner, repo, candidate.number)
    truths = []
    for path, size in candidate_files(
        tree, max_files=max_files, max_bytes=max_bytes, changed=changed
    ):
        quoted = urllib.parse.quote(path)
        touches = client.get(
            f"{API_ROOT}/repos/{owner}/{repo}/commits?path={quoted}&per_page={window}"
        )
        history = touches if isinstance(touches, list) else []
        _, agent = classify_touches(history, window=window)
        addition_lines = (
            _additions(client, owner, repo, commit_sha(history[0]), path)
            if agent and history
            else 0
        )
        # The current length bounds an attributed count and supplies the human side of the
        # fraction, so it is read for every examined file from the content CDN, which does
        # not spend the API budget the per-file history just spent.
        truths.append(
            file_truth(
                path,
                history,
                window=window,
                blob_bytes=size,
                additions=addition_lines,
                current_lines=client.line_count(owner, repo, branch, path),
            )
        )
    return RepositoryTruth(
        url=candidate.repository,
        pull_request=pull_request,
        branch=branch,
        tree_truncated=bool(tree.get("truncated")),
        files=tuple(truths),
        requests=client.budget.spent - spent_before,
        raw_requests=client.budget.raw_spent - raw_before,
    )


# Requests spent to read one candidate's pull request, repository, and file tree before its
# share is known. The payloads are kept, so the harvest of a ranked repository reads them again
# from memory instead of spending these requests twice.
RANK_REQUESTS = 3


def _remember(client: GithubClient, url: str, store: dict[str, Any]) -> Any:
    """Read a payload once and keep it for the harvest of the same repository."""
    payload = client.get(url)
    store[url] = payload
    return payload


def _take(client: GithubClient, url: str, store: dict[str, Any]) -> Any:
    """Return a payload the ranking already read, or read it now."""
    if url in store:
        return store.pop(url)
    return client.get(url)


def changed_share(
    client: GithubClient, candidate: MergeCandidate, store: dict[str, Any]
) -> float | None:
    """Return the fraction of a repository's files one merged pull request changed.

    Attribution is last-touch per file, so a pull request that changed most of a repository is
    the one that can attribute most of it. ``None`` means the repository could not be read.
    """
    owner, repo = parse_github_url(candidate.repository)
    if not owner or not repo:
        return None
    pull = _remember(client, f"{API_ROOT}/repos/{owner}/{repo}/pulls/{candidate.number}", store)
    meta = _remember(client, f"{API_ROOT}/repos/{owner}/{repo}", store)
    if not isinstance(pull, dict) or not isinstance(meta, dict):
        return None
    branch = str(meta.get("default_branch") or "main")
    tree = _remember(
        client, f"{API_ROOT}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1", store
    )
    if not isinstance(tree, dict):
        return None
    blobs = sum(1 for entry in tree.get("tree", []) if entry.get("type") == "blob")
    changed = int(pull.get("changed_files") or 0)
    if not blobs:
        return None
    return min(1.0, changed / blobs)


def rank_candidates(
    client: GithubClient,
    candidates: Iterable[MergeCandidate],
    known: set[str],
    store: dict[str, Any],
) -> list[MergeCandidate]:
    """Order unseen candidates so the repositories an agent changed most are harvested first.

    Candidates whose repository is already in the cohort are dropped before any request is
    spent on them. The sort is stable, so candidates that tie keep the search order.
    """
    scored: list[tuple[float, MergeCandidate]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.repository in known or candidate.repository in seen:
            continue
        seen.add(candidate.repository)
        if client.budget.remaining < RANK_REQUESTS:
            break
        share = changed_share(client, candidate, store)
        scored.append((share or 0.0, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored]


def harvest(
    client: GithubClient,
    *,
    agents: tuple[AgentApp, ...] = AGENT_APPS,
    since: str,
    until: str,
    per_agent_limit: int,
    max_repositories: int,
    store: CohortStore,
    window: int = TOUCH_WINDOW,
    max_files: int = 40,
    max_bytes: int = MAX_FILE_BYTES,
    refresh: bool = False,
) -> HarvestSummary:
    """Search for merged agent pull requests and harvest each repository once."""
    cohort = read_cohort(store.path)
    known = set() if refresh else cohort.urls()
    summary = HarvestSummary(repositories=len(cohort.repositories))
    prefetched: dict[str, Any] = {}
    print(
        f"[i] cohort {store.path}: {len(cohort.repositories)} repository(ies) and "
        f"{len(cohort.humans)} verified human repository(ies) already recorded"
    )
    for agent in agents:
        if summary.repositories >= max_repositories or summary.stopped:
            break
        print(f"[i] searching {agent.name} ({agent.slug}) merged pull requests {since}..{until}")
        try:
            candidates, partitions = search_merged(
                client, agent, since=since, until=until, limit=per_agent_limit
            )
        except BudgetExceeded as exc:
            summary.stopped = str(exc)
            break
        except RateLimited as exc:
            summary.stopped = str(exc)
            summary.retry_after = exc.retry_after
            break
        summary.partitions.extend(partitions)
        print(
            f"[i] {agent.slug}: {len(candidates)} candidate(s) from "
            f"{len(partitions)} partition(s), {client.budget.spent} request(s) spent"
        )
        if client.budget.remaining < RANK_REQUESTS:
            summary.stopped = (
                f"request ceiling reached ({client.budget.spent} of "
                f"{client.budget.ceiling} spent); the cohort keeps every repository "
                "that finished"
            )
            break
        try:
            ranked = rank_candidates(client, candidates, known, prefetched)
        except BudgetExceeded as exc:
            summary.stopped = str(exc)
            break
        except RateLimited as exc:
            summary.stopped = str(exc)
            summary.retry_after = exc.retry_after
            break
        seen: set[str] = set()
        for candidate in ranked:
            if summary.repositories >= max_repositories:
                summary.stopped = f"repository goal reached ({max_repositories})"
                break
            if candidate.repository in known or candidate.repository in seen:
                continue
            seen.add(candidate.repository)
            if client.budget.remaining < 3 + max_files:
                summary.stopped = (
                    f"request ceiling reached ({client.budget.spent} of "
                    f"{client.budget.ceiling} spent); the cohort keeps every repository "
                    "that finished"
                )
                break
            try:
                truth = harvest_repository(
                    client, candidate, window=window, max_files=max_files, max_bytes=max_bytes,
                    prefetched=prefetched,
                )
            except BudgetExceeded as exc:
                summary.stopped = str(exc)
                break
            except RateLimited as exc:
                summary.stopped = str(exc)
                summary.retry_after = exc.retry_after
                break
            if truth.rejected:
                summary.skipped += 1
                print(f"[!] {truth.url}: {truth.rejected}")
                continue
            store.add_repository(truth)
            known.add(truth.url)
            summary.repositories += 1
            if not truth.files_with(AGENT_FILE):
                # The pull request is real evidence of agent work, so the repository keeps its
                # label, but none of the files this run examined shows that agent's last touch.
                # The record says so, and a reader must not count it as attributed code.
                author = agent_for_slug(truth.pull_request.agent) if truth.pull_request else None
                if author is not None and not author.commit_attributable:
                    print(
                        f"[!] {truth.url}: {author.name} commits carry no agent identity, so "
                        "its files cannot be attributed from commits; only the pull request "
                        "and its merge commit are recorded"
                    )
                else:
                    print(
                        f"[!] {truth.url}: the agent pull request is recorded, but no "
                        "examined file was last touched by it; the repository carries no "
                        "file-level truth"
                    )
            fraction = truth.attributed_fraction
            shown = "n/a" if fraction is None else f"{fraction * 100:.1f}%"
            print(
                f"[i] {truth.url}: {len(truth.files_with(AGENT_FILE))} agent file(s) / "
                f"{len(truth.files_with(HUMAN_FILE))} human file(s), {shown} attributed "
                f"over {truth.lines_with(AGENT_FILE) + truth.lines_with(HUMAN_FILE)} "
                f"line(s), {truth.requests} request(s)"
            )
    summary.errors = list(client.errors[:20])
    return summary


__all__ = [
    "LANGUAGE_EXTENSIONS",
    "MAX_FILE_BYTES",
    "HarvestSummary",
    "candidate_files",
    "harvest",
    "harvest_repository",
    "language_of",
]
