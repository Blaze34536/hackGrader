"""Reject a candidate pool that cannot satisfy every quota before allocating.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..validation import REQUIRED_SOURCE_TYPES, REQUIRED_SPLIT_SOURCE_TYPES
from .bands import (
    _CLASS_DESCRIPTIONS,
    _CLASS_LABELS,
    _PROXIMITY_CLASSES,
    _SCORE_BANDS,
    _SCORE_CLASSES,
)
from .components import ComponentProfile
from .quotas import PartitionQuotas


@dataclass
class CapacityReport:
    """Which categories the pool must cover and what the allocator may draw on."""

    required_categories: set[str]
    required_category_partitions: dict[str, set[str]]
    category_components: dict[str, list[int]]
    real_repository_components: int
    real_repository_prompts: int
    real_repository_class_prompts: dict[str, int]


def category_index(
    profiles: list[ComponentProfile], required_categories: set[str]
) -> dict[str, list[int]]:
    """List the component indices that satisfy each required category."""
    return {
        category: [
            index
            for index, profile in enumerate(profiles)
            if category in profile.categories
        ]
        for category in required_categories
    }


def required_category_partitions(
    quotas: PartitionQuotas,
) -> dict[str, set[str]]:
    """Map each required category to the partitions whose quotas need it.

    Artifact examples are required in the combined pool but are linked augmentations of
    AIGCodeSet samples, so they are never required in a particular partition.
    """
    required = {
        category: set(quotas.partition_names)
        for category in REQUIRED_SPLIT_SOURCE_TYPES | set(_SCORE_CLASSES)
    }
    for label in _PROXIMITY_CLASSES:
        partitions_needing_boundary = {
            name
            for name in quotas.partition_names
            if quotas.class_minimums[name][label] > 0
        }
        if partitions_needing_boundary:
            required[label] = partitions_needing_boundary
    for band in _SCORE_BANDS:
        partitions_needing_band = {
            name
            for name in quotas.partition_names
            if quotas.band_minimums[name][band] > 0
        }
        if band in {"0-39", "60-100"}:
            partitions_needing_band.update(quotas.partition_names)
        if partitions_needing_band:
            required[f"score_band:{band}"] = partitions_needing_band
    return required


def require_capacity(
    rows: list[dict[str, Any]],
    profiles: list[ComponentProfile],
    quotas: PartitionQuotas,
    total_unique_prompts: int,
) -> CapacityReport:
    """Report every way the pool falls short together, then fail closed."""
    partition_names = quotas.partition_names
    issues: list[str] = []

    source_types = {row["source_type"] for row in rows}
    missing_source_types = REQUIRED_SOURCE_TYPES - source_types
    if missing_source_types:
        issues.append(
            f"candidate pool is missing source families: {sorted(missing_source_types)}"
        )
    if len(profiles) == 1:
        issues.append(
            "all examples share source material; cannot create leakage-free partitions"
        )
    elif len(profiles) < len(partition_names):
        issues.append(
            "not enough distinct source groups for leakage-free partitions: "
            f"need {len(partition_names)}, found {len(profiles)}"
        )

    needed_partitions = required_category_partitions(quotas)
    required_categories = set(needed_partitions)
    category_components = category_index(profiles, required_categories)
    for category in sorted(category_components):
        indices = category_components[category]
        partitions = needed_partitions[category]
        if len(indices) < len(partitions):
            issues.append(
                f"not enough source-disjoint groups for {category!r}: "
                f"need {len(partitions)}, found {len(indices)}"
            )
        if category.startswith("score_band:"):
            band = category.removeprefix("score_band:")
            expected = sum(quotas.band_minimums[name][band] for name in partitions)
            available = sum(profiles[index].bands[band] for index in indices)
            if available < expected:
                issues.append(
                    f"not enough distinct prompts in the {band} score band for split "
                    f"quotas: need {expected}, found {available}"
                )

    required_prompt_count = sum(quotas.requested_rows.values())
    if total_unique_prompts < required_prompt_count:
        issues.append(
            "not enough distinct prompts for configured split sizes: need "
            f"{required_prompt_count}, found {total_unique_prompts}"
        )

    for label in _CLASS_LABELS:
        available = sum(profile.classes[label] for profile in profiles)
        expected = sum(quotas.class_minimums[name][label] for name in partition_names)
        if available < expected:
            issues.append(
                f"not enough distinct {_CLASS_DESCRIPTIONS[label]} prompts for split "
                f"quotas: need {expected}, found {available}"
            )

    repository_components = sum(profile.has_repository for profile in profiles)
    real_repository_prompts = sum(
        len(profile.real_repository_prompts) for profile in profiles
    )
    real_repository_class_prompts = {
        label: sum(profile.real_repository_classes[label] for profile in profiles)
        for label in _CLASS_LABELS
    }
    required_repository_components = sum(quotas.real_repository_minimums.values())
    if repository_components < required_repository_components:
        issues.append(
            "not enough source-disjoint real-repository components for split quotas: "
            f"need {required_repository_components} "
            f"({quotas.real_repository_minimums} by split), "
            f"found {repository_components}"
        )
    required_repository_prompts = sum(
        quotas.real_repository_prompt_minimums.values()
    )
    if real_repository_prompts < required_repository_prompts:
        issues.append(
            "not enough distinct real-repository prompts for split quotas: "
            f"need {required_repository_prompts}, found {real_repository_prompts}"
        )
    for label in _SCORE_CLASSES:
        required = sum(
            quotas.real_repository_class_minimums[name][label]
            for name in partition_names
        )
        if real_repository_class_prompts[label] < required:
            issues.append(
                f"not enough real-repository {label}-range prompts for split quotas: "
                f"need {required}, found {real_repository_class_prompts[label]}"
            )

    if issues:
        raise ValueError("; ".join(issues))
    return CapacityReport(
        required_categories=required_categories,
        required_category_partitions=needed_partitions,
        category_components=category_components,
        real_repository_components=repository_components,
        real_repository_prompts=real_repository_prompts,
        real_repository_class_prompts=real_repository_class_prompts,
    )
