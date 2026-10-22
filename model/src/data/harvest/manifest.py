"""Derive the fail-closed repository label manifest from a harvested cohort.

The harvester does not invent a label format. It writes what
:mod:`data.labels` already validates — an ``ai`` entry carrying the merged agent pull request
and its merge commit, and a ``human`` entry carrying the protocol that ruled the class in —
so the manifest it produces is accepted or rejected by the same standard the candidate
builder applies, and every entry it cannot support is left out rather than softened.

Entries keep the measurement they were derived from as extra fields (files attributed, lines
attributed, rule version), because the next reader's first question about a label is which
files it rests on. Those fields are ignored by the validator and recorded for the auditor.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..labels import (
    AGENT_CONTRIBUTION,
    CONTROLLED_NO_AI_ALLOWED,
    LABEL_AI,
    LABEL_HUMAN,
    LabelManifest,
    report_manifest,
    validate_entries,
)
from .bands import entry_band
from .records import (
    AGENT_FILE,
    ATTRIBUTION_RULE,
    HUMAN_FILE,
    Cohort,
    HumanVerification,
    RepositoryTruth,
)

AUDITOR = "cohort-harvester"


def file_measurements(truth: RepositoryTruth) -> list[dict[str, Any]]:
    """Return each attributed file's AI and examined line counts, as the repository share is taken.

    Each file's examined count is its current line count, so its share is the fraction of its own
    lines the agent wrote. A batch's share is then the fraction over the lines of the files it shows.
    This is not the repository's ``examined_lines``: that total counts an agent file by its attributed
    lines alone, so a per-file share would always be 0 or 100 and could not give a batch a middle value.
    Unknown files are left out, the same way the repository share leaves them out.
    """
    return [
        {
            "path": item.path,
            "ai_lines": item.ai_lines,
            "examined_lines": item.current_lines,
        }
        for item in truth.files
        if item.attribution in (AGENT_FILE, HUMAN_FILE)
    ]


def ai_entry(
    truth: RepositoryTruth, *, audited_by: str = AUDITOR, audited_at: str
) -> dict[str, Any] | None:
    """Return the ``ai`` manifest entry for a harvested repository, or ``None``.

    A repository is only labelable when its agent authorship has a commit to point at. The
    merge commit is that pointer, so a repository whose pull request record is missing one is
    left out of the manifest even though its files were attributed — an entry the validator
    would reject anyway, and one that should not be written as if it passed.
    """
    pull_request = truth.pull_request
    if pull_request is None or not pull_request.merge_commit:
        return None
    return {
        "url": truth.url,
        "label": LABEL_AI,
        "evidence": [
            {
                "evidence_class": AGENT_CONTRIBUTION,
                "agent": pull_request.agent,
                "pull_request": pull_request.url,
                "commit": pull_request.merge_commit,
            }
        ],
        "audited_by": audited_by,
        "audited_at": audited_at,
        "attribution_rule": ATTRIBUTION_RULE,
        "attributed_files": len(truth.files_with(AGENT_FILE)),
        "examined_files": len(truth.files),
        "attributed_lines": truth.lines_with(AGENT_FILE),
        "examined_lines": truth.lines_with(AGENT_FILE) + truth.lines_with(HUMAN_FILE),
        "merged_at": pull_request.merged_at,
        "files": file_measurements(truth),
    }


def human_entry(
    verification: HumanVerification,
    *,
    audited_by: str = AUDITOR,
    audited_at: str,
) -> dict[str, Any]:
    """Return the ``human`` manifest entry for a verified repository."""
    return {
        "url": verification.url,
        "label": LABEL_HUMAN,
        "evidence": [
            {
                "evidence_class": CONTROLLED_NO_AI_ALLOWED,
                "protocol": verification.protocol,
            }
        ],
        "audited_by": audited_by,
        "audited_at": audited_at,
        "commits_examined": verification.commits_examined,
        "newest_commit": verification.newest_commit,
    }


def derive_entries(
    cohort: Cohort, *, audited_by: str = AUDITOR, audited_at: str
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Return manifest entries for a cohort beside the repositories that had to be left out."""
    entries: list[dict[str, Any]] = []
    dropped: list[tuple[str, str]] = []
    for truth in cohort.repositories:
        entry = ai_entry(truth, audited_by=audited_by, audited_at=audited_at)
        if entry is None:
            dropped.append((truth.url, "no merge commit recorded for the agent pull request"))
            continue
        entries.append(entry)
    for verification in cohort.humans:
        entries.append(human_entry(verification, audited_by=audited_by, audited_at=audited_at))
    return entries, dropped


def write_manifest(path: str, entries: list[dict[str, Any]]) -> str:
    """Write manifest entries as JSON Lines and return the path written."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def report(path: str, entries: list[dict[str, Any]]) -> LabelManifest:
    """Validate entries with the shared label contract and print what it accepted.

    Reporting through ``data.labels`` means the harvester cannot claim a manifest the builder
    would refuse: the accepted and rejected counts printed here are the counts the data build
    will see, and a rejection is shown with the standard's own reason.
    """
    accepted, rejected = validate_entries(entries)
    manifest = LabelManifest(path, accepted, rejected)
    report_manifest(manifest)
    return manifest


__all__ = [
    "AUDITOR",
    "ai_entry",
    "balance_entries",
    "derive_entries",
    "human_entry",
    "report",
    "write_manifest",
]


def balance_entries(
    entries: list[dict[str, Any]], *, goal: int
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Keep at most ``goal`` entries per band, in order, and drop AI entries outside every band."""
    kept: list[dict[str, Any]] = []
    dropped: list[tuple[str, str]] = []
    counts: dict[str, int] = {}
    for entry in entries:
        band = entry_band(entry)
        if band is None:
            dropped.append((entry["url"], "attributed share falls outside every AI band"))
            continue
        if counts.get(band, 0) >= goal:
            dropped.append((entry["url"], f"the {band} band already holds {goal}"))
            continue
        counts[band] = counts.get(band, 0) + 1
        kept.append(entry)
    return kept, dropped
