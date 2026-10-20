"""Plan agent-by-month search partitions and read merged agent pull requests from them.

A search returns at most 1,000 results however many exist, so one query per agent is not a
cohort: it is the first thousand of millions, all from one afternoon. Partitions are
therefore agent by calendar month, halved further when a month still exceeds the cap, and each
partition is recorded with the exact query string it ran and the result count it claimed.
That record is what makes the cohort reproducible rather than merely asserted.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import calendar
import urllib.parse
from dataclasses import dataclass
from datetime import date, timedelta

from .agents import AgentApp
from .client import API_ROOT, GithubClient
from .records import Partition

SEARCH_URL = f"{API_ROOT}/search/issues"
MAX_SEARCH_RESULTS = 1000
PAGE_SIZE = 100
MAX_PAGES = MAX_SEARCH_RESULTS // PAGE_SIZE


@dataclass(frozen=True)
class MergeCandidate:
    """One merged agent pull request, before its repository has been read."""

    repository: str
    repository_api: str
    number: int
    agent: str
    author: str
    merged_at: str

    @property
    def pull_request_url(self) -> str:
        """Return the pull request URL recorded as this candidate's evidence."""
        return f"{self.repository}/pull/{self.number}"


def _parse(value: str) -> date:
    """Parse an ISO date, refusing anything else with a readable error."""
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise SystemExit(f"[!] {value!r} is not an ISO date (expected YYYY-MM-DD).") from None


def month_windows(since: str, until: str) -> list[tuple[str, str]]:
    """Return calendar-month partitions covering an inclusive date range."""
    start, end = _parse(since), _parse(until)
    if start > end:
        raise SystemExit(f"[!] --since {since} is after --until {until}.")
    windows: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        last_day = calendar.monthrange(cursor.year, cursor.month)[1]
        stop = min(end, date(cursor.year, cursor.month, last_day))
        windows.append((cursor.isoformat(), stop.isoformat()))
        cursor = stop + timedelta(days=1)
    return windows


def halve(window: tuple[str, str]) -> list[tuple[str, str]]:
    """Split one window in two, so a result cap can be worked around by a narrower query."""
    start, end = _parse(window[0]), _parse(window[1])
    if start >= end:
        return [window]
    middle = start + timedelta(days=(end - start).days // 2)
    return [
        (start.isoformat(), middle.isoformat()),
        ((middle + timedelta(days=1)).isoformat(), end.isoformat()),
    ]


def partition_query(agent: AgentApp, window: tuple[str, str]) -> str:
    """Return the search query for one agent and date window."""
    return f"is:pr is:merged author:app/{agent.slug} merged:{window[0]}..{window[1]}"


def _page(client: GithubClient, query: str, page: int) -> dict:
    """Return one page of search results for a query."""
    url = (
        f"{SEARCH_URL}?q={urllib.parse.quote(query)}&per_page={PAGE_SIZE}"
        f"&page={page}&sort=created&order=desc"
    )
    payload = client.get(url)
    return payload if isinstance(payload, dict) else {}


def _candidate(item: dict, agent: AgentApp) -> MergeCandidate | None:
    """Convert one search item into a repository candidate, or ``None`` when unusable."""
    repository_api = str(item.get("repository_url") or "")
    repository = str(item.get("html_url") or "").rsplit("/pull/", 1)[0]
    if not repository_api or "/pull/" not in str(item.get("html_url") or ""):
        return None
    author = item.get("user") or {}
    merged = item.get("pull_request") or {}
    return MergeCandidate(
        repository=repository,
        repository_api=repository_api,
        number=int(item.get("number") or 0),
        agent=agent.slug,
        author=str(author.get("login") or ""),
        merged_at=str(merged.get("merged_at") or ""),
    )


def search_merged(
    client: GithubClient,
    agent: AgentApp,
    *,
    since: str,
    until: str,
    limit: int,
    max_depth: int = 3,
) -> tuple[list[MergeCandidate], list[Partition]]:
    """Return merged pull requests for one agent and the partitions they were found through.

    A window whose result count exceeds the 1,000-result cap is recorded once with
    ``capped`` set and then replaced by two half-windows, so the provenance shows both the
    query that revealed the cap and the narrower queries that worked around it. A window
    still over the cap at ``max_depth`` is reported as capped rather than as complete.
    """
    candidates: list[MergeCandidate] = []
    partitions: list[Partition] = []
    pending: list[tuple[tuple[str, str], int]] = [
        (window, 0) for window in month_windows(since, until)
    ]
    while pending and len(candidates) < limit:
        window, depth = pending.pop(0)
        query = partition_query(agent, window)
        payload = _page(client, query, 1)
        total = int(payload.get("total_count") or 0)
        if total > MAX_SEARCH_RESULTS and depth < max_depth:
            partitions.append(
                Partition(
                    agent=agent.slug,
                    query=query,
                    since=window[0],
                    until=window[1],
                    total_count=total,
                    pages=1,
                    capped=True,
                )
            )
            pending[0:0] = [(part, depth + 1) for part in halve(window)]
            continue
        pages = min(MAX_PAGES, -(-total // PAGE_SIZE))
        fetched = 0
        read_pages = 0
        for page in range(1, pages + 1):
            page_payload = payload if page == 1 else _page(client, query, page)
            items = page_payload.get("items") or []
            read_pages += 1
            for item in items:
                if len(candidates) >= limit:
                    break
                candidate = _candidate(item, agent)
                if candidate is None:
                    continue
                candidates.append(candidate)
                fetched += 1
            if not items or len(candidates) >= limit:
                break
        partitions.append(
            Partition(
                agent=agent.slug,
                query=query,
                since=window[0],
                until=window[1],
                total_count=total,
                fetched=fetched,
                pages=read_pages,
                capped=total > MAX_SEARCH_RESULTS,
            )
        )
    return candidates, partitions


__all__ = [
    "MAX_SEARCH_RESULTS",
    "MergeCandidate",
    "halve",
    "month_windows",
    "partition_query",
    "search_merged",
]
