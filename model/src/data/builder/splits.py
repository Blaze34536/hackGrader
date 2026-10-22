"""Turn a candidate pool into verified, source-disjoint training partitions.

The order matters: conflicts are removed globally before anything is deduplicated or
capped, so a prompt that appears with two different targets cannot survive in one split
and leak into another through a shared source group.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import os
import random
from typing import Any

import contracts as report
from learning.config import (
    MIN_BOUNDARY_CLASS_ROWS,
    MIN_GENERATED_TEST_BOUNDARY_CLASS_ROWS,
    MIN_GENERATED_TEST_CLASS_ROWS,
    MIN_GENERATED_TEST_MIDDLE_SCORE_BAND_ROWS,
    MIN_GENERATED_TEST_ROWS,
    MIN_GENERATED_TEST_SCORE_BAND_ROWS,
    MIN_SCORE_BAND_ROWS,
    MIN_TRAIN_CLASS_ROWS,
    MIN_TRAIN_MIDDLE_SCORE_BAND_ROWS,
    MIN_TRAIN_ROWS,
    MIN_VALIDATION_BOUNDARY_CLASS_ROWS,
    MIN_VALIDATION_CLASS_ROWS,
    MIN_VALIDATION_MIDDLE_SCORE_BAND_ROWS,
    MIN_VALIDATION_ROWS,
    MIN_VALIDATION_SCORE_BAND_ROWS,
)
from learning.preflight import validate_dataset_sizes

from ..config import (
    MIN_REAL_REPOSITORIES_PER_SPLIT,
    TEST_FRACTION,
    VALIDATION_FRACTION,
)
from ..rows import cap_targets, distinct, quarantine_conflicting_prompts
from ..splitting import split_by_source_with_test
from ..splitting.exclusion import exclude_repositories
from ..verification import verify_splits


def _deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Merge rows that share a rendered prompt, family by family, and count them."""
    pool: list[dict[str, Any]] = []
    duplicates = 0
    families = sorted({(row["source_kind"], row["source_type"]) for row in rows})
    for source_kind, source_type in families:
        family_rows, removed = distinct(
            [
                row
                for row in rows
                if row["source_kind"] == source_kind
                and row["source_type"] == source_type
            ]
        )
        pool.extend(family_rows)
        duplicates += removed
    return pool, duplicates


def split_candidate_rows(
    pool: list[dict[str, Any]],
    rng: random.Random,
    *,
    fresh_repository_test: bool = False,
    min_real_repository_prompts: int = 50,
    min_real_repository_class_rows: int = 10,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Partition a pool with every configured floor, explaining an infeasible one."""
    real_repository_minimums = MIN_REAL_REPOSITORIES_PER_SPLIT
    quotas = {}
    if fresh_repository_test:
        real_repository_minimums = {
            "training": MIN_REAL_REPOSITORIES_PER_SPLIT,
            "validation": MIN_REAL_REPOSITORIES_PER_SPLIT,
            "test": MIN_REAL_REPOSITORIES_PER_SPLIT,
        }
        quotas = {
            "min_real_repository_prompts": {
                "test": min_real_repository_prompts
            },
            "min_real_repository_class_rows": {
                "test": {
                    "human": min_real_repository_class_rows,
                    "ai": min_real_repository_class_rows,
                }
            },
        }
    try:
        return split_by_source_with_test(
            pool,
            VALIDATION_FRACTION,
            TEST_FRACTION,
            rng,
            report_contract=report,
            min_real_repositories_per_split=real_repository_minimums,
            **quotas,
            min_score_band_rows={
                "training": MIN_SCORE_BAND_ROWS,
                "validation": MIN_VALIDATION_SCORE_BAND_ROWS,
                "test": MIN_GENERATED_TEST_SCORE_BAND_ROWS,
            },
            min_middle_score_band_rows={
                "training": MIN_TRAIN_MIDDLE_SCORE_BAND_ROWS,
                "validation": MIN_VALIDATION_MIDDLE_SCORE_BAND_ROWS,
                "test": MIN_GENERATED_TEST_MIDDLE_SCORE_BAND_ROWS,
            },
            min_rows={
                "training": MIN_TRAIN_ROWS,
                "validation": MIN_VALIDATION_ROWS,
                "test": MIN_GENERATED_TEST_ROWS,
            },
            min_class_rows={
                "training": MIN_TRAIN_CLASS_ROWS,
                "validation": MIN_VALIDATION_CLASS_ROWS,
                "test": MIN_GENERATED_TEST_CLASS_ROWS,
            },
            min_boundary_class_rows={
                "training": MIN_BOUNDARY_CLASS_ROWS,
                "validation": MIN_VALIDATION_BOUNDARY_CLASS_ROWS,
                "test": MIN_GENERATED_TEST_BOUNDARY_CLASS_ROWS,
            },
        )
    except ValueError as exc:
        source_code = getattr(split_by_source_with_test, "__code__", None)
        splitter_path = os.path.abspath(
            getattr(source_code, "co_filename", "unknown")
        )
        raise SystemExit(
            f"[!] Cannot produce a source-disjoint split: {exc} "
            f"(splitter: {splitter_path})"
        ) from None


def split_rows(
    rows: list[dict[str, Any]], args, rng: random.Random
) -> tuple[list[dict], list[dict], list[dict]]:
    """Hold out benchmarks, quarantine target conflicts, deduplicate, cap, split, and gate."""
    rows = exclude_repositories(rows, getattr(args, "exclude_repositories", None))
    rows, quarantined = quarantine_conflicting_prompts(rows)
    if quarantined:
        print(
            f"[i] Removed {quarantined} conflicting or source-linked candidate "
            "row(s) before capping and source-disjoint splitting."
        )
    pool, duplicates = _deduplicate(rows)
    if duplicates:
        print(f"[i] Merged provenance for {duplicates} duplicate rendered prompt(s).")
    pool = cap_targets(pool, args.target_cap, rng)
    if not pool:
        raise SystemExit("No training rows were produced; nothing to write.")

    train_rows, val_rows, test_rows = split_candidate_rows(
        pool,
        rng,
        fresh_repository_test=getattr(args, "fresh_repository_test", False),
        min_real_repository_prompts=getattr(
            args, "min_fresh_test_repository_prompts", 50
        ),
        min_real_repository_class_rows=getattr(
            args, "min_fresh_test_class_rows", 10
        ),
    )
    if any(row["source_type"] == "real_repository" for row in pool):
        originals = [
            row for row in train_rows if row["source_type"] == "real_repository"
        ]
        train_rows.extend(originals * max(0, args.repo_repeat - 1))
        rng.shuffle(train_rows)

    try:
        validate_dataset_sizes(train_rows, val_rows, test_rows, report)
    except ValueError as exc:
        raise SystemExit(
            f"[!] Candidate data failed the training size gates: {exc}"
        ) from None
    try:
        verify_splits(
            [
                ("training", train_rows),
                ("validation", val_rows),
                ("independent test", test_rows),
            ],
            report,
            require_full_sources=True,
        )
    except ValueError as exc:
        raise SystemExit(
            f"[!] Candidate data failed a provenance/holdout gate: {exc}"
        ) from None
    return train_rows, val_rows, test_rows
