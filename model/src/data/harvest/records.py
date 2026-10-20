"""Cohort records: per-file attribution truth, provenance, and the restart-safe store.

The cohort exists because a repository authorship label has to be recomputable by someone
else. Each record therefore carries what it takes to recheck it: the pull request and merge
commit an agent authored, the commit SHA that last touched every file examined, the rule
version that turned those commits into an attribution, and the requests spent asking.

The store is append-only JSON Lines. A harvest that is interrupted leaves every finished
repository behind, and a restart skips what is already recorded instead of paying for it
again; a torn final line is dropped on read rather than failing the run.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

# Both are recorded in every record, because a truth computed under one rule version must
# never be read as if a later rule produced it.
HARVESTER_VERSION = "harvest-1"
ATTRIBUTION_RULE = "last-touch-v1"

REPOSITORY_KIND = "repository"
HUMAN_KIND = "human"

AGENT_FILE = "agent"
HUMAN_FILE = "human"
UNKNOWN_FILE = "unknown"


@dataclass(frozen=True)
class Partition:
    """One agent-by-date search window and what asking it cost."""

    agent: str
    query: str
    since: str
    until: str
    total_count: int = 0
    fetched: int = 0
    pages: int = 0
    capped: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the partition as JSON-ready fields."""
        return {
            "agent": self.agent,
            "query": self.query,
            "since": self.since,
            "until": self.until,
            "total_count": self.total_count,
            "fetched": self.fetched,
            "pages": self.pages,
            "capped": self.capped,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Partition:
        """Rebuild a partition from a recorded dictionary."""
        return cls(
            agent=str(data.get("agent", "")),
            query=str(data.get("query", "")),
            since=str(data.get("since", "")),
            until=str(data.get("until", "")),
            total_count=int(data.get("total_count", 0)),
            fetched=int(data.get("fetched", 0)),
            pages=int(data.get("pages", 0)),
            capped=bool(data.get("capped", False)),
        )


@dataclass(frozen=True)
class PullRequest:
    """The merged agent pull request that made a repository part of the cohort."""

    url: str
    number: int
    agent: str
    author: str
    merged_at: str
    merge_commit: str

    def to_dict(self) -> dict[str, Any]:
        """Return the pull request as JSON-ready fields."""
        return {
            "url": self.url,
            "number": self.number,
            "agent": self.agent,
            "author": self.author,
            "merged_at": self.merged_at,
            "merge_commit": self.merge_commit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PullRequest | None:
        """Rebuild a pull request record, or return ``None`` when none was recorded."""
        if not data:
            return None
        return cls(
            url=str(data.get("url", "")),
            number=int(data.get("number", 0)),
            agent=str(data.get("agent", "")),
            author=str(data.get("author", "")),
            merged_at=str(data.get("merged_at", "")),
            merge_commit=str(data.get("merge_commit", "")),
        )


@dataclass(frozen=True)
class FileTruth:
    """One examined file and the commit that last changed it.

    ``additions`` and ``current_lines`` are kept apart from the attribution so a reader can
    recompute the bounded count instead of trusting it: the attributed line count is the
    lines the last-touch commit added, capped by the length the file has now, because a
    commit that added 40 lines to a file since rewritten to 12 did not leave 40 lines of AI
    code behind.
    """

    path: str
    attribution: str
    touches_examined: int
    blob_bytes: int = 0
    last_touch_sha: str = ""
    last_touch_agent: str = ""
    last_touch_author: str = ""
    last_touch_date: str = ""
    additions: int = 0
    current_lines: int = 0

    @property
    def ai_lines(self) -> int:
        """Return the attributed line count, zero for a file no agent last touched."""
        if self.attribution != AGENT_FILE:
            return 0
        return min(self.additions, self.current_lines)

    def to_dict(self) -> dict[str, Any]:
        """Return the file truth as JSON-ready fields."""
        return {
            "path": self.path,
            "attribution": self.attribution,
            "touches_examined": self.touches_examined,
            "blob_bytes": self.blob_bytes,
            "last_touch_sha": self.last_touch_sha,
            "last_touch_agent": self.last_touch_agent,
            "last_touch_author": self.last_touch_author,
            "last_touch_date": self.last_touch_date,
            "additions": self.additions,
            "current_lines": self.current_lines,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileTruth:
        """Rebuild a file truth from a recorded dictionary."""
        return cls(
            path=str(data.get("path", "")),
            attribution=str(data.get("attribution", UNKNOWN_FILE)),
            touches_examined=int(data.get("touches_examined", 0)),
            blob_bytes=int(data.get("blob_bytes", 0)),
            last_touch_sha=str(data.get("last_touch_sha", "")),
            last_touch_agent=str(data.get("last_touch_agent", "")),
            last_touch_author=str(data.get("last_touch_author", "")),
            last_touch_date=str(data.get("last_touch_date", "")),
            additions=int(data.get("additions", 0)),
            current_lines=int(data.get("current_lines", 0)),
        )


@dataclass(frozen=True)
class RepositoryTruth:
    """One agent-active repository: its files, their attribution, and what the read cost."""

    url: str
    pull_request: PullRequest | None
    branch: str = ""
    tree_truncated: bool = False
    files: tuple[FileTruth, ...] = ()
    requests: int = 0
    raw_requests: int = 0
    rejected: str = ""

    def files_with(self, attribution: str) -> tuple[FileTruth, ...]:
        """Return the examined files carrying one attribution."""
        return tuple(item for item in self.files if item.attribution == attribution)

    def lines_with(self, attribution: str) -> int:
        """Return the attributed line count over the examined files of one class."""
        if attribution == AGENT_FILE:
            return sum(item.ai_lines for item in self.files)
        return sum(item.current_lines for item in self.files_with(attribution))

    @property
    def attributed_fraction(self) -> float | None:
        """Return the AI share over the examined lines, or ``None`` when nothing was read.

        This is the fraction over the files the harvest examined, not over the repository:
        the label manifest and the candidate builder scale it by the coverage the reader
        reports, which is why the examined and attributed line counts are both recorded
        rather than only their ratio.
        """
        agent_lines = self.lines_with(AGENT_FILE)
        total = agent_lines + self.lines_with(HUMAN_FILE)
        return agent_lines / total if total else None

    def to_dict(self) -> dict[str, Any]:
        """Return the repository truth as JSON-ready fields."""
        return {
            "kind": REPOSITORY_KIND,
            "version": HARVESTER_VERSION,
            "rule": ATTRIBUTION_RULE,
            "url": self.url,
            "branch": self.branch,
            "tree_truncated": self.tree_truncated,
            "pull_request": self.pull_request.to_dict() if self.pull_request else None,
            "files": [item.to_dict() for item in self.files],
            "requests": self.requests,
            "raw_requests": self.raw_requests,
            "rejected": self.rejected,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepositoryTruth:
        """Rebuild a repository truth from a recorded dictionary."""
        return cls(
            url=str(data.get("url", "")),
            pull_request=PullRequest.from_dict(data.get("pull_request")),
            branch=str(data.get("branch", "")),
            tree_truncated=bool(data.get("tree_truncated", False)),
            files=tuple(FileTruth.from_dict(item) for item in data.get("files", [])),
            requests=int(data.get("requests", 0)),
            raw_requests=int(data.get("raw_requests", 0)),
            rejected=str(data.get("rejected", "")),
        )


@dataclass(frozen=True)
class HumanVerification:
    """A repository verified as the human class by a recorded no-agent protocol."""

    url: str
    protocol: str
    commits_examined: int
    newest_commit: str = ""
    branch: str = ""
    requests: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the verification as JSON-ready fields."""
        return {
            "kind": HUMAN_KIND,
            "version": HARVESTER_VERSION,
            "rule": ATTRIBUTION_RULE,
            "url": self.url,
            "protocol": self.protocol,
            "commits_examined": self.commits_examined,
            "newest_commit": self.newest_commit,
            "branch": self.branch,
            "requests": self.requests,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanVerification:
        """Rebuild a human verification from a recorded dictionary."""
        return cls(
            url=str(data.get("url", "")),
            protocol=str(data.get("protocol", "")),
            commits_examined=int(data.get("commits_examined", 0)),
            newest_commit=str(data.get("newest_commit", "")),
            branch=str(data.get("branch", "")),
            requests=int(data.get("requests", 0)),
        )


@dataclass
class Cohort:
    """Every repository record read back from one cohort file."""

    path: str
    repositories: list[RepositoryTruth] = field(default_factory=list)
    humans: list[HumanVerification] = field(default_factory=list)

    def urls(self) -> set[str]:
        """Return every repository URL the cohort already covers."""
        return {item.url for item in (*self.repositories, *self.humans)}

    def counts(self) -> dict[str, int]:
        """Count the cohort by record kind."""
        return {"repository": len(self.repositories), "human": len(self.humans)}


def _json_line(record: dict[str, Any]) -> str:
    """Serialize one record as a single JSON line."""
    return json.dumps(record, sort_keys=True) + "\n"


def read_cohort(path: str) -> Cohort:
    """Read a cohort file, dropping a torn final line rather than failing the run."""
    cohort = Cohort(path)
    if not os.path.exists(path):
        return cohort
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # An interrupted write leaves a partial line. Every complete record before
                # it is still valid harvest output, so the tail is dropped and the harvest
                # resumes from the last repository it finished.
                continue
            if not isinstance(record, dict):
                continue
            if record.get("kind") == HUMAN_KIND:
                cohort.humans.append(HumanVerification.from_dict(record))
            elif record.get("kind") == REPOSITORY_KIND:
                cohort.repositories.append(RepositoryTruth.from_dict(record))
    return cohort


class CohortStore:
    """Append cohort records durably, so an interrupted harvest loses no finished work."""

    def __init__(self, path: str) -> None:
        """Append to a cohort file, creating it on the first record."""
        self.path = path
        self.repositories = 0
        self.humans = 0

    def _append(self, record: dict[str, Any]) -> None:
        """Append one record and flush it to the operating system before returning."""
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(_json_line(record))
            handle.flush()
            os.fsync(handle.fileno())

    def add_repository(self, truth: RepositoryTruth) -> None:
        """Record one harvested repository."""
        self._append(truth.to_dict())
        self.repositories += 1

    def add_human(self, verification: HumanVerification) -> None:
        """Record one verified human-class repository."""
        self._append(verification.to_dict())
        self.humans += 1


def provenance_path(path: str) -> str:
    """Return the provenance sidecar path for a cohort file."""
    root, _ = os.path.splitext(path)
    return f"{root}.provenance.json"


def write_provenance(path: str, run: dict[str, Any]) -> str:
    """Append one run's provenance to the sidecar and return the path written.

    Provenance is a sidecar rather than a header line because the cohort file is append-only
    and restarted: a header would have to be rewritten in place on every run, while the
    budget, partitions, and rate-limit floor measured at the end of a run only exist once
    the run is over. Each run stays in ``runs``, so what a cohort was built from is
    reproducible after several sessions.
    """
    target = provenance_path(path)
    runs: list[dict[str, Any]] = []
    if os.path.exists(target):
        try:
            with open(target, encoding="utf-8") as handle:
                existing = json.load(handle)
            if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
                runs = [item for item in existing["runs"] if isinstance(item, dict)]
        except (OSError, json.JSONDecodeError):
            runs = []
    runs.append(run)
    payload = {
        "cohort": os.path.abspath(path),
        "harvester": HARVESTER_VERSION,
        "rule": ATTRIBUTION_RULE,
        "runs": runs,
    }
    temporary = f"{target}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)
    return target
