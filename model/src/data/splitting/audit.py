"""Check each allocated partition against every quota before it is used.

The allocator reasons about supply while it places components, so a defect in that
bookkeeping shows up as a partition that silently misses a floor. Re-measuring the
finished partitions from the rows themselves is the cheapest way to catch it.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from ..validation import (
    MAX_TARGET_SHARE,
    REQUIRED_SPLIT_SOURCE_TYPES,
    canonical_source_url,
)
from .bands import (
    _CLASS_DESCRIPTIONS,
    _CLASS_LABELS,
    _SCORE_BANDS,
    _SCORE_CLASSES,
    class_labels,
    score_band,
)
from .components import ComponentProfile
from .quotas import PartitionQuotas


def build_partitions(
    rows: list[dict[str, Any]],
    profiles: list[ComponentProfile],
    assigned: dict[int, str],
    quotas: PartitionQuotas,
) -> dict[str, list[dict[str, Any]]]:
    """Collect each partition's rows in component order, training as the default."""
    return {
        name: [
            rows[row_index]
            for index, profile in enumerate(profiles)
            if assigned.get(index, "training") == name
            for row_index in profile.rows
        ]
        for name in quotas.partition_names
    }


def _partition_prompts(
    rows: list[dict[str, Any]],
    partitions: dict[str, list[dict[str, Any]]],
    prompt_scores: dict[str, int],
) -> dict[str, dict[str, int]]:
    """Return the distinct prompt targets each partition actually contains."""
    return {
        name: {
            row["messages"][1]["content"]: prompt_scores[row["messages"][1]["content"]]
            for row in partition
        }
        for name, partition in partitions.items()
    }


def _verify_partition(
    name: str,
    partition: list[dict[str, Any]],
    prompts: dict[str, int],
    quotas: PartitionQuotas,
) -> None:
    """Raise if one partition misses any floor it was allocated against."""
    if not partition:
        raise ValueError(f"source-grouped split produced an empty {name} partition")
    if len(prompts) < quotas.requested_rows[name]:
        raise ValueError(
            f"{name} partition has {len(prompts)} distinct prompts; "
            f"at least {quotas.requested_rows[name]} are required"
        )

    class_counts = {
        label: sum(label in class_labels(score) for score in prompts.values())
        for label in _CLASS_LABELS
    }
    for label in _CLASS_LABELS:
        if class_counts[label] < quotas.class_minimums[name][label]:
            raise ValueError(
                f"{name} partition does not meet its {_CLASS_DESCRIPTIONS[label]} "
                f"minimum: has {class_counts[label]}, "
                f"needs {quotas.class_minimums[name][label]}"
            )

    bands = Counter(score_band(score) for score in prompts.values())
    if any(bands[band] < quotas.band_minimums[name][band] for band in _SCORE_BANDS):
        raise ValueError(
            f"{name} partition does not meet its score-band minimums: {dict(bands)}"
        )

    repositories = {
        canonical_source_url(row["source_url"])
        for row in partition
        if row["source_type"] == "real_repository"
    }
    if len(repositories) < quotas.real_repository_minimums[name]:
        raise ValueError(
            f"{name} partition contains {len(repositories)} independent real "
            f"repositories; at least {quotas.real_repository_minimums[name]} are required"
        )
    repository_prompts = {
        row["messages"][1]["content"]
        for row in partition
        if row["source_type"] == "real_repository"
    }
    if len(repository_prompts) < quotas.real_repository_prompt_minimums[name]:
        raise ValueError(
            f"{name} partition has {len(repository_prompts)} real-repository prompts; "
            f"at least {quotas.real_repository_prompt_minimums[name]} are required"
        )
    repository_scores = {
        prompt: prompts[prompt] for prompt in repository_prompts if prompt in prompts
    }
    repository_class_counts = {
        label: sum(label in class_labels(score) for score in repository_scores.values())
        for label in _SCORE_CLASSES
    }
    for label, count in repository_class_counts.items():
        minimum = quotas.real_repository_class_minimums[name][label]
        if count < minimum:
            raise ValueError(
                f"{name} partition has {count} real-repository {label}-range prompts; "
                f"at least {minimum} are required"
            )
    missing_types = REQUIRED_SPLIT_SOURCE_TYPES - {
        row["source_type"] for row in partition
    }
    if missing_types:
        raise ValueError(
            f"{name} partition is missing source families: {sorted(missing_types)}"
        )
    if ({"human", "ai"} - {label for label, count in class_counts.items() if count}):
        raise ValueError(f"{name} partition is missing one of the score classes")

    coverage_bands = {"0-39", "60-100"} | {
        band for band in _SCORE_BANDS if quotas.band_minimums[name][band] > 0
    }
    if missing_bands := coverage_bands - set(bands):
        raise ValueError(
            f"{name} partition is missing required score bands: {sorted(missing_bands)}"
        )

    dominant_score, dominant_count = Counter(prompts.values()).most_common(1)[0]
    if dominant_count / len(prompts) > MAX_TARGET_SHARE:
        raise ValueError(
            f"{name} partition has {dominant_count / len(prompts):.0%} of targets at "
            f"{dominant_score}%; the largest single score share is {MAX_TARGET_SHARE:.0%}"
        )


def verify_partitions(
    rows: list[dict[str, Any]],
    profiles: list[ComponentProfile],
    assigned: dict[int, str],
    prompt_scores: dict[str, int],
    quotas: PartitionQuotas,
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    """Verify every partition, then shuffle rows so component order is not visible."""
    partitions = build_partitions(rows, profiles, assigned, quotas)
    prompts = _partition_prompts(rows, partitions, prompt_scores)
    for name, partition in partitions.items():
        _verify_partition(name, partition, prompts[name], quotas)
        rng.shuffle(partition)
    return partitions
