"""Verify split lineage, message contracts, and target integrity.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .validation import (
    MAX_TARGET_SHARE,
    REQUIRED_SOURCE_TYPES,
    REQUIRED_SPLIT_SOURCE_TYPES,
    canonical_source_url,
    validate_row,
)


def verify_splits(
    partitions: list[tuple[str, list[dict[str, Any]]]],
    report_contract: Any,
    *,
    require_full_sources: bool = True,
    require_score_diversity: bool = True,
) -> None:
    """Reject invalid targets, missing source families, and pairwise split leakage."""
    if len(partitions) < 2:
        raise ValueError("at least two data splits are required")
    names = [name for name, _ in partitions]
    if len(names) != len(set(names)):
        raise ValueError("data split names must be unique")
    if any(not rows for _, rows in partitions):
        raise ValueError("all data splits must contain examples")

    source_groups_by_split: dict[str, set[str]] = {}
    source_urls_by_split: dict[str, set[str]] = {}
    prompts_by_split: dict[str, set[str]] = {}
    targets_by_split: dict[str, dict[str, int]] = {}
    source_types_by_split: dict[str, set[str]] = {}
    all_source_types: set[str] = set()

    for split_name, rows in partitions:
        source_groups: set[str] = set()
        source_urls: set[str] = set()
        unique_targets: dict[str, int] = {}
        source_types: set[str] = set()
        for index, row in enumerate(rows):
            validate_row(row, f"{split_name} data", index, report_contract)
            prompt = row["messages"][1]["content"]
            score = report_contract.parse_report(row["messages"][-1]["content"])[
                "ai_score"
            ]
            if score is None or not 0 <= score <= 100:
                raise ValueError(f"{split_name} row {index} has an unparseable AI target")
            previous_score = unique_targets.setdefault(prompt, score)
            if previous_score != score:
                raise ValueError(
                    f"{split_name} repeats a prompt with conflicting AI targets"
                )
            source_groups.update(
                group.strip().casefold() for group in row["source_groups"]
            )
            source_urls.add(canonical_source_url(row["source_url"]))
            source_types.add(row["source_type"])
            all_source_types.add(row["source_type"])

        source_groups_by_split[split_name] = source_groups
        source_urls_by_split[split_name] = source_urls
        prompts_by_split[split_name] = set(unique_targets)
        targets_by_split[split_name] = unique_targets
        source_types_by_split[split_name] = source_types

    for first_index, (first_name, _) in enumerate(partitions):
        for second_name, _ in partitions[first_index + 1 :]:
            if overlap := source_groups_by_split[first_name] & source_groups_by_split[second_name]:
                raise ValueError(
                    f"{len(overlap)} underlying source group(s) cross the "
                    f"{first_name}/{second_name} split"
                )
            if overlap := source_urls_by_split[first_name] & source_urls_by_split[second_name]:
                raise ValueError(
                    f"{len(overlap)} source URL(s) cross the {first_name}/{second_name} split"
                )
            if overlap := prompts_by_split[first_name] & prompts_by_split[second_name]:
                raise ValueError(
                    f"{len(overlap)} identical user prompt(s) cross the "
                    f"{first_name}/{second_name} split"
                )

    if require_full_sources:
        for split_name, rows in partitions:
            missing = REQUIRED_SPLIT_SOURCE_TYPES - source_types_by_split[split_name]
            if missing:
                raise ValueError(
                    f"{split_name} data is missing required independent source families "
                    f"({', '.join(sorted(REQUIRED_SPLIT_SOURCE_TYPES))}): "
                    f"{sorted(missing)}"
                )
            if "real_repository" in REQUIRED_SPLIT_SOURCE_TYPES and not any(
                row["source_type"] == "real_repository" for row in rows
            ):
                raise ValueError(f"{split_name} data contains no real-repository evidence")
            scores = targets_by_split[split_name].values()
            if not any(score < 40 for score in scores) or not any(
                score >= 60 for score in scores
            ):
                raise ValueError(
                    f"{split_name} scores must include a human-range target below 40% "
                    "and an AI-range target of at least 60%"
                )
        missing = REQUIRED_SOURCE_TYPES - all_source_types
        if missing:
            raise ValueError(
                f"combined data is missing required source families: {sorted(missing)}"
            )

    if require_score_diversity:
        for split_name, targets in targets_by_split.items():
            target, count = Counter(targets.values()).most_common(1)[0]
            share = count / len(targets)
            if share > MAX_TARGET_SHARE:
                raise ValueError(
                    f"{share:.0%} of {split_name} examples target {target}%; "
                    "targets are too concentrated"
                )


def verify_split(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    report_contract: Any,
    *,
    require_full_sources: bool = True,
    require_score_diversity: bool = True,
    reference_rows: list[dict[str, Any]] | None = None,
) -> None:
    """Check two named partitions and, optionally, an additional reference partition."""
    partitions = [("training", train_rows), ("validation", validation_rows)]
    if reference_rows is not None:
        partitions.append(("reference", reference_rows))
    verify_splits(
        partitions,
        report_contract,
        require_full_sources=require_full_sources,
        require_score_diversity=require_score_diversity,
    )
