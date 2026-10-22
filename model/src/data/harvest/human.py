"""Verify a human-class repository against a protocol that can be rechecked.

The human class cannot be established by looking for missing evidence: nobody finding an
agent commit is not the same as no agent having written the code. What a repository can
supply is a *recorded protocol* that rules the class in — every commit predates LLM-assisted
coding, and no registered agent appears among the commits of its default branch — so the
label states what was checked and the check can be repeated by someone else from the same
commit list.

Repositories are candidates for this class, not members of it. A repository that fails any
part of the protocol is rejected with its reason, because a pool of ordinary projects is
exactly where an agent-assisted one hides, and a fail-closed label means the failing case is
recorded rather than quietly dropped.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import urllib.parse
from datetime import date, datetime

from repositories.urls import parse_github_url

from .agents import AgentApp
from .attribution import commit_agent, commit_date
from .budget import BudgetExceeded
from .client import API_ROOT, GithubClient
from .records import HumanVerification

COMMIT_SAMPLE = 100


def _day(stamp: str) -> date | None:
    """Return the date part of an ISO timestamp, or ``None`` when it is unreadable."""
    text = str(stamp or "").strip()
    if not text:
        return None
    for parser in (date.fromisoformat, lambda value: datetime.fromisoformat(value).date()):
        try:
            return parser(text)
        except ValueError:
            continue
    return None


def verify_human_repository(
    client: GithubClient,
    url: str,
    *,
    cutoff: str,
    sample: int = COMMIT_SAMPLE,
) -> tuple[HumanVerification | None, str]:
    """Return a verified human-class record, or ``None`` and the reason it was rejected.

    The protocol is: the repository's last push predates ``cutoff``, the newest commit on its
    default branch predates ``cutoff``, and no commit among the newest ``sample`` was authored
    by a registered agent app. A history longer than the sample is still accepted, because
    the recorded claim is about the commits examined, not about the ones that were never read.
    """
    owner, repo = parse_github_url(url)
    if not owner or not repo:
        return None, "URL could not be parsed"
    before = client.budget.spent
    meta = client.get(f"{API_ROOT}/repos/{owner}/{repo}")
    if not isinstance(meta, dict):
        return None, "repository metadata could not be read"
    branch = str(meta.get("default_branch") or "main")
    cutoff_day = _day(cutoff)
    if cutoff_day is None:
        raise SystemExit(f"[!] {cutoff!r} is not an ISO date (expected YYYY-MM-DD).")
    pushed = _day(str(meta.get("pushed_at") or meta.get("updated_at") or ""))
    if pushed is not None and pushed >= cutoff_day:
        return None, f"pushed {pushed.isoformat()}, which is not before {cutoff}"
    commits = client.get(f"{API_ROOT}/repos/{owner}/{repo}/commits?per_page={sample}")
    if not isinstance(commits, list) or not commits:
        return None, "commit history could not be read"
    for commit in commits:
        if not isinstance(commit, dict):
            continue
        found: AgentApp | None = commit_agent(commit)
        if found is not None:
            return None, f"authored by a registered agent ({found.slug})"
    newest = _day(commit_date(commits[0] if isinstance(commits[0], dict) else {}))
    if newest is None:
        return None, "the newest commit has no readable date"
    if newest >= cutoff_day:
        return None, f"newest commit {newest.isoformat()} is not before {cutoff}"
    protocol = (
        f"every commit inspected predates {cutoff} (newest {newest.isoformat()}) and no "
        f"registered agent app appears among the {len(commits)} most recent commit author(s); "
        f"checked {date.today().isoformat()} on the default branch {branch}"
    )
    return (
        HumanVerification(
            url=url,
            protocol=protocol,
            commits_examined=len(commits),
            newest_commit=commit_date(commits[0]) if isinstance(commits[0], dict) else "",
            branch=branch,
            requests=client.budget.spent - before,
        ),
        "",
    )


def verify_human_repositories(
    client: GithubClient,
    urls: list[str],
    *,
    cutoff: str,
    sample: int = COMMIT_SAMPLE,
) -> tuple[list[HumanVerification], list[tuple[str, str]]]:
    """Verify every URL and return the accepted records beside the rejections."""
    accepted: list[HumanVerification] = []
    rejected: list[tuple[str, str]] = []
    for url in urls:
        if client.budget.remaining < 2:
            break
        try:
            verification, reason = verify_human_repository(
                client, url, cutoff=cutoff, sample=sample
            )
        except BudgetExceeded:
            break
        if verification is None:
            rejected.append((url, reason))
            print(f"[!] human class rejected {url}: {reason}")
        else:
            accepted.append(verification)
            print(f"[i] human class verified {url}: {verification.commits_examined} commit(s) inspected")
    return accepted, rejected


def read_url_list(path: str) -> list[str]:
    """Read repository URLs from a text file, ignoring blank lines and comments."""
    try:
        with open(path, encoding="utf-8") as handle:
            return [
                line.split("#", 1)[0].strip()
                for line in handle
                if line.split("#", 1)[0].strip()
            ]
    except OSError as exc:
        raise SystemExit(f"[!] Repository list {path} could not be read: {exc}") from None


__all__ = [
    "COMMIT_SAMPLE",
    "SEARCH_LANGUAGES",
    "read_url_list",
    "search_human_candidates",
    "verify_human_repositories",
    "verify_human_repository",
]


# The languages the agent cohort is mostly written in, so the human class is not told apart
# by language alone.
SEARCH_LANGUAGES = ("python", "javascript", "typescript", "go", "java")


def search_human_candidates(
    client: GithubClient,
    *,
    before: str,
    limit: int,
    languages: tuple[str, ...] = SEARCH_LANGUAGES,
) -> list[str]:
    """Return repository URLs last pushed before ``before``, as human-class candidates.

    The newest-pushed come first, so the human class is as close in time to the agent cohort
    as the protocol allows; every candidate still has to pass ``verify_human_repository``.
    """
    per_language = -(-limit // max(1, len(languages)))
    urls: list[str] = []
    for language in languages:
        query = f"pushed:<{before} language:{language} stars:>=10 fork:false"
        payload = client.get(
            f"{API_ROOT}/search/repositories?q={urllib.parse.quote(query)}"
            f"&sort=updated&order=desc&per_page={min(100, per_language)}"
        )
        items = payload.get("items", []) if isinstance(payload, dict) else []
        for item in items[:per_language]:
            url = str(item.get("html_url") or "")
            if url and not item.get("fork") and url not in urls:
                urls.append(url)
    return urls[:limit]
