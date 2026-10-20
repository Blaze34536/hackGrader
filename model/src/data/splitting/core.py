"""Plan and materialize source-disjoint train, validation, and test partitions.

Components of the split live in sibling modules: ``bands`` names the target classes,
``components`` groups rows by lineage, ``quotas`` resolves the floors to meet,
``capacity`` rejects a pool that cannot meet them, ``allocator`` places whole
components, and ``audit`` re-measures the finished partitions.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .allocator import ComponentAllocator
from .audit import verify_partitions
from .capacity import require_capacity
from .components import ComponentProfile, _components, profile_components, prompt_scores
from .quotas import PartitionQuotas, build_quotas


@dataclass
class SplitPlan:
    """The resolved quotas, profiled components, and where each component was placed."""

    quotas: PartitionQuotas
    profiles: list[ComponentProfile]
    assigned: dict[int, str]
    prompt_scores: dict[str, int]


def plan_partitions(
    rows: list[dict[str, Any]],
    holdout_fractions: tuple[tuple[str, float], ...],
    rng: random.Random,
    *,
    report_contract: Any,
    min_real_repositories_per_split: int | Mapping[str, int],
    min_score_band_rows: int | Mapping[str, int],
    min_middle_score_band_rows: int | Mapping[str, int] | None,
    min_rows: int | Mapping[str, int],
    min_class_rows: int | Mapping[str, int],
    min_boundary_class_rows: int | Mapping[str, int] = 0,
    min_real_repository_prompts: int | Mapping[str, int] = 0,
    min_real_repository_class_rows: int | Mapping[str, int | Mapping[str, int]] = 0,
) -> SplitPlan:
    """Resolve quotas, prove the pool can satisfy them, then reserve whole components."""
    if not rows:
        raise ValueError("cannot split an empty candidate pool")
    scores = prompt_scores(rows, report_contract)
    quotas = build_quotas(
        holdout_fractions,
        min_real_repositories_per_split=min_real_repositories_per_split,
        min_score_band_rows=min_score_band_rows,
        min_middle_score_band_rows=min_middle_score_band_rows,
        min_rows=min_rows,
        min_class_rows=min_class_rows,
        min_boundary_class_rows=min_boundary_class_rows,
        min_real_repository_prompts=min_real_repository_prompts,
        min_real_repository_class_rows=min_real_repository_class_rows,
        total_unique_prompts=len(scores),
    )
    profiles = profile_components(rows, _components(rows), scores)
    capacity = require_capacity(rows, profiles, quotas, len(scores))

    order = list(range(len(profiles)))
    rng.shuffle(order)
    profiles = [profiles[index] for index in order]
    assigned = ComponentAllocator(
        profiles, quotas, capacity, holdout_fractions
    ).run()
    return SplitPlan(
        quotas=quotas, profiles=profiles, assigned=assigned, prompt_scores=scores
    )


def partition_rows(
    rows: list[dict[str, Any]],
    holdout_fractions: tuple[tuple[str, float], ...],
    rng: random.Random,
    **options: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Split rows into named partitions, each verified against its resolved quotas."""
    plan = plan_partitions(rows, holdout_fractions, rng, **options)
    return verify_partitions(
        rows,
        plan.profiles,
        plan.assigned,
        plan.prompt_scores,
        plan.quotas,
        rng,
    )


def split_by_source(
    rows: list[dict[str, Any]],
    validation_fraction: float,
    rng: random.Random,
    *,
    report_contract: Any,
    min_real_repositories_per_split: int = 0,
    min_score_band_rows: int | Mapping[str, int] = 1,
    min_middle_score_band_rows: int | Mapping[str, int] | None = None,
    min_rows: int | Mapping[str, int] = 0,
    min_class_rows: int | Mapping[str, int] = 0,
    min_boundary_class_rows: int | Mapping[str, int] = 0,
    min_real_repository_prompts: int | Mapping[str, int] = 0,
    min_real_repository_class_rows: int | Mapping[str, int | Mapping[str, int]] = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split complete source groups into training and validation partitions."""
    partitions = partition_rows(
        rows,
        (("validation", validation_fraction),),
        rng,
        report_contract=report_contract,
        min_real_repositories_per_split=min_real_repositories_per_split,
        min_score_band_rows=min_score_band_rows,
        min_middle_score_band_rows=min_middle_score_band_rows,
        min_rows=min_rows,
        min_class_rows=min_class_rows,
        min_boundary_class_rows=min_boundary_class_rows,
        min_real_repository_prompts=min_real_repository_prompts,
        min_real_repository_class_rows=min_real_repository_class_rows,
    )
    return partitions["training"], partitions["validation"]


def split_by_source_with_test(
    rows: list[dict[str, Any]],
    validation_fraction: float,
    test_fraction: float,
    rng: random.Random,
    *,
    report_contract: Any,
    min_real_repositories_per_split: int = 0,
    min_score_band_rows: int | Mapping[str, int] = 1,
    min_middle_score_band_rows: int | Mapping[str, int] | None = None,
    min_rows: int | Mapping[str, int] = 0,
    min_class_rows: int | Mapping[str, int] = 0,
    min_boundary_class_rows: int | Mapping[str, int] = 0,
    min_real_repository_prompts: int | Mapping[str, int] = 0,
    min_real_repository_class_rows: int | Mapping[str, int | Mapping[str, int]] = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create train, validation, and final-test sets with disjoint source lineage."""
    partitions = partition_rows(
        rows,
        (("validation", validation_fraction), ("test", test_fraction)),
        rng,
        report_contract=report_contract,
        min_real_repositories_per_split=min_real_repositories_per_split,
        min_score_band_rows=min_score_band_rows,
        min_middle_score_band_rows=min_middle_score_band_rows,
        min_rows=min_rows,
        min_class_rows=min_class_rows,
        min_boundary_class_rows=min_boundary_class_rows,
        min_real_repository_prompts=min_real_repository_prompts,
        min_real_repository_class_rows=min_real_repository_class_rows,
    )
    return partitions["training"], partitions["validation"], partitions["test"]
