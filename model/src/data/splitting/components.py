"""Group rows by shared lineage and measure what each component can supply.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..validation import canonical_source_url, validate_row
from .bands import _CLASS_LABELS, _SCORE_BANDS, class_labels, score_band


def _components(rows: list[dict[str, Any]]) -> list[list[int]]:
    """Group rows sharing URLs, source lineage, or rendered prompts."""
    parents = list(range(len(rows)))
    identity_owner: dict[str, int] = {}

    def root(index: int) -> int:
        """Find a component representative while compressing the parent chain."""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(first: int, second: int) -> None:
        """Merge two row components that share provenance or a prompt."""
        first_root = root(first)
        second_root = root(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for index, row in enumerate(rows):
        prompt = (
            row["messages"][1]["content"] if "messages" in row else row["prompt"]
        )
        identities = {
            f"url:{canonical_source_url(row.get('source_url', row.get('url', '')))}",
            f"prompt:{prompt}",
        }
        identities.update(
            f"source:{group.strip().casefold()}"
            for group in row.get("source_groups", [])
        )
        for identity in identities:
            previous = identity_owner.setdefault(identity, index)
            join(index, previous)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[root(index)].append(index)
    return list(groups.values())


@dataclass
class ComponentProfile:
    """Everything one indivisible lineage component contributes to a split."""

    rows: list[int]
    prompts: dict[str, int]
    types: set[str] = field(default_factory=set)
    repositories: set[str] = field(default_factory=set)
    classes: dict[str, int] = field(default_factory=dict)
    real_repository_prompts: dict[str, int] = field(default_factory=dict)
    real_repository_classes: dict[str, int] = field(default_factory=dict)
    bands: dict[str, int] = field(default_factory=dict)
    categories: set[str] = field(default_factory=set)

    @property
    def prompt_count(self) -> int:
        """Return how many distinct prompts the component carries."""
        return len(self.prompts)

    @property
    def has_repository(self) -> bool:
        """Return whether the component carries independent real-repository rows."""
        return bool(self.repositories)


def prompt_scores(rows: list[dict[str, Any]], report_contract: Any) -> dict[str, int]:
    """Validate candidate rows and return each prompt's single agreed target score."""
    scores: dict[str, int] = {}
    for index, row in enumerate(rows):
        validate_row(row, "candidate data", index, report_contract)
        prompt = row["messages"][1]["content"]
        score = report_contract.parse_report(row["messages"][-1]["content"])["ai_score"]
        if score is None:
            raise ValueError(f"candidate row {index} has no usable AI target")
        previous_score = scores.setdefault(prompt, score)
        if previous_score != score:
            raise ValueError("candidate data repeats a prompt with conflicting AI targets")
    return scores


def profile_components(
    rows: list[dict[str, Any]],
    components: list[list[int]],
    scores: dict[str, int],
) -> list[ComponentProfile]:
    """Measure each component's prompt count, classes, bands, and provenance."""
    profiles: list[ComponentProfile] = []
    for component in components:
        prompts = {
            rows[index]["messages"][1]["content"]: scores[
                rows[index]["messages"][1]["content"]
            ]
            for index in component
        }
        types = {rows[index]["source_type"] for index in component}
        repositories = {
            canonical_source_url(rows[index]["source_url"])
            for index in component
            if rows[index]["source_type"] == "real_repository"
        }
        classes = {label: 0 for label in _CLASS_LABELS}
        for score in prompts.values():
            for label in class_labels(score):
                classes[label] += 1
        real_repository_prompts = {
            rows[index]["messages"][1]["content"]: scores[
                rows[index]["messages"][1]["content"]
            ]
            for index in component
            if rows[index]["source_type"] == "real_repository"
        }
        real_repository_classes = {label: 0 for label in _CLASS_LABELS}
        for score in real_repository_prompts.values():
            for label in class_labels(score):
                real_repository_classes[label] += 1
        bands = {
            band: sum(score_band(score) == band for score in prompts.values())
            for band in _SCORE_BANDS
        }
        categories = set(types)
        categories.update(label for label in _CLASS_LABELS if classes[label])
        categories.update(
            f"score_band:{band}" for band, count in bands.items() if count
        )
        profiles.append(
            ComponentProfile(
                rows=list(component),
                prompts=prompts,
                types=types,
                repositories=repositories,
                classes=classes,
                real_repository_prompts=real_repository_prompts,
                real_repository_classes=real_repository_classes,
                bands=bands,
                categories=categories,
            )
        )
    return profiles
