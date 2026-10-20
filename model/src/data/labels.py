"""Fail-closed repository authorship labels and their provenance score bands.

A repository row may only enter a candidate with a label whose evidence can be
re-derived from an external, auditable record. A README that claims an AI wrote the
project is not that record, and neither is the detector's own style reading: wiring
either one in as supervision is what produced repository targets that contradicted
their provenance. Entries that cannot meet the standard are rejected rather than
silently relabelled.

Manifest format (JSON Lines, one repository per line)::

    {"url": "https://github.com/owner/repo",
     "label": "ai",
     "evidence": [{"evidence_class": "agent_contribution",
                   "agent": "devin",
                   "pull_request": "https://github.com/owner/repo/pull/12",
                   "commit": "0f1e2d3c4b5a69788796a5b4c3d2e1f001234567"}],
     "audited_by": "reviewer-name",
     "audited_at": "2026-10-12"}

An ``ai`` entry needs an auditable AI contribution to source code: an agent-attributed
pull request merged into the repository, or a recorded generation trace. A ``human``
entry needs a controlled collection in which AI use was prohibited and its protocol
recorded, because the absence of an observed agent contribution does not prove that
nobody used AI.

An entry may also carry the measurement its label rests on — ``attributed_lines`` and
``examined_lines``, or ``attributed_files`` and ``examined_files``. When it does, the
row's target is that percentage rather than the midpoint of its band, so the model learns
the share the record established instead of a class constant. The harvester writes those
counts on every AI entry it derives.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from .config import AI_LABEL_BAND, ARTIFACT_BAND, FLAG_THRESHOLD, HUMAN_LABEL_BAND
from .features import has_explicit_artifact
from .validation import canonical_source_url

LABEL_AI = "ai"
LABEL_HUMAN = "human"
LABELS = (LABEL_AI, LABEL_HUMAN)

AGENT_CONTRIBUTION = "agent_contribution"
GENERATION_TRACE = "generation_trace"
CONTROLLED_NO_AI_ALLOWED = "controlled_no_ai_allowed"
AUDITABLE_AI_CLASSES = (AGENT_CONTRIBUTION, GENERATION_TRACE)
EVIDENCE_CLASSES = (*AUDITABLE_AI_CLASSES, CONTROLLED_NO_AI_ALLOWED)

# A repository that merely says an AI wrote it is a claim, not a record. The auditor
# still records those declarations so a rejected pool member can be told apart from an
# entry that supplied no evidence at all.
SELF_DECLARED_CLASSES = ("self_declared_readme", "readme_declaration", "self_declared")

PROVENANCE_BANDS = {
    LABEL_AI: AI_LABEL_BAND,
    LABEL_HUMAN: HUMAN_LABEL_BAND,
}

__all__ = [
    "AGENT_CONTRIBUTION",
    "AUDITABLE_AI_CLASSES",
    "CONTROLLED_NO_AI_ALLOWED",
    "EVIDENCE_CLASSES",
    "GENERATION_TRACE",
    "LABELS",
    "LABEL_AI",
    "LABEL_HUMAN",
    "PROVENANCE_BANDS",
    "SELF_DECLARED_CLASSES",
    "LabelManifest",
    "RejectedLabel",
    "load_repository_labels",
    "measured_share",
    "provenance_target",
    "report_manifest",
    "require_usable",
    "validate_entries",
]


@dataclass(frozen=True)
class RejectedLabel:
    """One manifest entry that failed the fail-closed evidence standard."""

    index: int
    url: str
    reason: str


@dataclass
class LabelManifest:
    """Accepted repository labels keyed by canonical URL, plus every rejection."""

    path: str
    accepted: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected: list[RejectedLabel] = field(default_factory=list)

    def label_for(self, url: str) -> str | None:
        """Return the accepted label for a repository URL, if the manifest has one."""
        entry = self.entry_for(url)
        return entry["label"] if entry else None

    def entry_for(self, url: str) -> dict[str, Any] | None:
        """Return the accepted entry for a repository URL, with its recorded measurement."""
        return self.accepted.get(canonical_source_url(url))

    def counts(self) -> dict[str, int]:
        """Count accepted labels by class."""
        counts = {label: 0 for label in LABELS}
        for entry in self.accepted.values():
            counts[entry["label"]] += 1
        return counts


def _is_github_url(url: Any) -> bool:
    """Return whether a value is a credential-free GitHub repository URL."""
    if not isinstance(url, str) or not url.strip():
        return False
    parsed = urlsplit(url if "://" in url else f"https://{url}")
    if parsed.username or parsed.password:
        return False
    parts = [part for part in parsed.path.split("/") if part]
    return (
        (parsed.hostname or "").casefold() in {"github.com", "www.github.com"}
        and len(parts) >= 2
    )


def _text(entry: dict[str, Any], name: str) -> str:
    """Return a trimmed string field, or an empty string when it is absent."""
    value = entry.get(name)
    return value.strip() if isinstance(value, str) else ""


def _ai_evidence_reason(records: list[dict[str, Any]]) -> str | None:
    """Return why no auditable AI contribution record is present, or ``None``."""
    for record in records:
        if record.get("evidence_class") == AGENT_CONTRIBUTION and (
            _text(record, "pull_request") and _text(record, "commit")
        ):
            return None
        if record.get("evidence_class") == GENERATION_TRACE and _text(record, "trace"):
            return None
    return (
        "an ai label needs an agent-attributed pull request with its merged commit, "
        "or a recorded generation trace"
    )


def _human_evidence_reason(records: list[dict[str, Any]]) -> str | None:
    """Return why no controlled no-AI record is present, or ``None``."""
    for record in records:
        if (
            record.get("evidence_class") == CONTROLLED_NO_AI_ALLOWED
            and _text(record, "protocol")
        ):
            return None
    return (
        "a human label needs a controlled collection that prohibited AI use and "
        "recorded its protocol; a missing agent contribution is not proof"
    )


def _entry_reason(entry: Any) -> str | None:
    """Return the first reason an entry fails the standard, or ``None`` when it passes."""
    if not isinstance(entry, dict):
        return "entry is not a JSON object"
    if not _is_github_url(entry.get("url")):
        return "url must be a GitHub repository URL without credentials"
    label = entry.get("label")
    if label not in LABELS:
        return f"label must be one of {sorted(LABELS)}"
    if not _text(entry, "audited_by") or not _text(entry, "audited_at"):
        return "entry needs audited_by and audited_at"
    records = entry.get("evidence")
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(record, dict) for record in records)
    ):
        return "evidence must be a non-empty list of records"
    classes = {record.get("evidence_class") for record in records}
    unknown = classes - set(EVIDENCE_CLASSES) - set(SELF_DECLARED_CLASSES)
    if unknown:
        return f"unknown evidence class(es): {sorted(map(str, unknown))}"
    ai_reason = _ai_evidence_reason(records)
    human_reason = _human_evidence_reason(records)
    if ai_reason is None and human_reason is None:
        return "entry mixes AI-contribution and controlled-no-AI evidence"
    return ai_reason if label == LABEL_AI else human_reason


def validate_entries(entries: list[Any]) -> tuple[dict[str, dict[str, Any]], list[RejectedLabel]]:
    """Validate manifest entries, rejecting every duplicate and unsupported label."""
    keys = [
        canonical_source_url(entry["url"])
        for entry in entries
        if isinstance(entry, dict) and _is_github_url(entry.get("url"))
    ]
    duplicates = {key for key, count in Counter(keys).items() if count > 1}
    accepted: dict[str, dict[str, Any]] = {}
    rejected: list[RejectedLabel] = []
    for index, entry in enumerate(entries):
        url = entry.get("url") if isinstance(entry, dict) else ""
        key = canonical_source_url(url) if isinstance(url, str) else ""
        reason = _entry_reason(entry)
        if reason is None and key in duplicates:
            reason = "duplicate repository URL in the manifest"
        if reason is not None:
            rejected.append(RejectedLabel(index, str(url), reason))
            continue
        accepted[key] = entry
    return accepted, rejected


def load_repository_labels(path: str) -> LabelManifest:
    """Read and fail-closed validate a JSON Lines repository label manifest."""
    entries: list[Any] = []
    try:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    raise SystemExit(
                        f"[!] {path} line {number} is not valid JSON; no labels loaded."
                    ) from None
    except OSError as exc:
        raise SystemExit(
            f"[!] Repository label manifest {path} could not be read: {exc}"
        ) from None
    accepted, rejected = validate_entries(entries)
    return LabelManifest(path, accepted, rejected)


def measured_share(entry: dict[str, Any] | None) -> float | None:
    """Return the attributed percentage a manifest entry recorded, or ``None``.

    Lines are read before files: the target answers what share of the code was written
    elsewhere, and a project with three generated one-line helpers beside one hand-written
    thousand-line module is not 75% agent-written. A measurement that is malformed or
    contradicts itself is ignored rather than trusted, so a bad count cannot silently
    become supervision.
    """
    if not isinstance(entry, dict):
        return None
    for attributed_name, examined_name in (
        ("attributed_lines", "examined_lines"),
        ("attributed_files", "examined_files"),
    ):
        attributed = entry.get(attributed_name)
        examined = entry.get(examined_name)
        if isinstance(attributed, bool) or isinstance(examined, bool):
            continue
        if not isinstance(attributed, (int, float)) or not isinstance(examined, (int, float)):
            continue
        if examined <= 0 or attributed < 0 or attributed > examined:
            continue
        return 100.0 * attributed / examined
    return None


def provenance_target(
    label: str, analysis: dict[str, Any], entry: dict[str, Any] | None = None
) -> int | None:
    """Return the target a verified label assigns, or ``None`` when contradicted.

    Provenance outranks the detector's style reading, because the whole point of a
    verified label is that the style reading is what needs correcting. The target is the
    percentage the manifest *measured* — a repository whose agent-attributed lines are 4%
    of its code teaches 4, and one at 96% teaches 96 — not the midpoint of a band. The band
    midpoint survives only for an entry that records a class and no measurement, which is a
    coarser claim and is the reason the harvester writes the counts on every entry.

    A verified human-class repository teaches 0 even when the manifest records no counts:
    the protocol that puts it in that class is that no agent wrote any of it, so a band
    midpoint would teach a share the label denies.

    Two contradictions drop the row instead of being supervised: a human-labeled
    repository carrying an explicit AI-generation marker, and an ``ai`` label whose own
    recorded attribution is zero. A measured human-class repository above the flag
    threshold contradicts its protocol the same way.
    """
    if label not in PROVENANCE_BANDS:
        raise ValueError(f"unknown repository label {label!r}")
    if has_explicit_artifact(analysis):
        if label == LABEL_HUMAN:
            return None
        low, high = ARTIFACT_BAND
        return round((low + high) / 2)
    share = measured_share(entry)
    if share is not None:
        target = round(share)
        if label == LABEL_AI and target <= 0:
            return None
        if label == LABEL_HUMAN and target >= FLAG_THRESHOLD:
            return None
        return target
    if label == LABEL_HUMAN:
        return 0
    low, high = PROVENANCE_BANDS[label]
    return round((low + high) / 2)


def report_manifest(manifest: LabelManifest) -> None:
    """Print accepted and rejected counts with the leading rejection reasons."""
    counts = manifest.counts()
    print(
        f"[i] repository labels from {manifest.path}: {len(manifest.accepted)} accepted "
        f"({counts[LABEL_AI]} ai, {counts[LABEL_HUMAN]} human), "
        f"{len(manifest.rejected)} rejected"
    )
    for item in manifest.rejected[:8]:
        print(f"    rejected {item.url or '<missing url>'}: {item.reason}")
    if len(manifest.rejected) > 8:
        print(f"    ... and {len(manifest.rejected) - 8} more rejected entr(ies)")


def require_usable(manifest: LabelManifest) -> None:
    """Stop a build whose verified labels cannot supervise both classes."""
    if not manifest.accepted:
        raise SystemExit(
            "[!] No repository label passed the fail-closed evidence standard; "
            "nothing can be supervised from provenance."
        )
    counts = manifest.counts()
    if not counts[LABEL_AI] or not counts[LABEL_HUMAN]:
        raise SystemExit(
            "[!] Verified repository labels must cover both classes; found "
            f"{counts[LABEL_AI]} ai and {counts[LABEL_HUMAN]} human."
        )
