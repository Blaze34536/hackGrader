"""Expand and validate every per-partition quota a split has to satisfy.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .bands import _PROXIMITY_CLASSES, _SCORE_BANDS, _SCORE_CLASSES


def _minimums(
    value: int | Mapping[str, int],
    partition_names: list[str],
    *,
    name: str,
    default: int,
) -> dict[str, int]:
    """Expand a shared minimum or partial per-partition mapping."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer or a partition mapping")
    if isinstance(value, int):
        minimums = {partition: value for partition in partition_names}
    elif isinstance(value, Mapping):
        unknown = set(value) - set(partition_names)
        if unknown:
            raise ValueError(f"{name} contains unknown partitions: {sorted(unknown)}")
        minimums = {
            partition: value.get(partition, default) for partition in partition_names
        }
    else:
        raise ValueError(f"{name} must be an integer or a partition mapping")
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in minimums.values()
    ):
        raise ValueError(f"{name} values must be non-negative integers")
    return minimums


@dataclass
class PartitionQuotas:
    """Every floor one split must clear, resolved per partition."""

    partition_names: list[str]
    holdout_fractions: tuple[tuple[str, float], ...]
    min_real_repositories_per_split: int | dict[str, int]
    row_minimums: dict[str, int]
    class_minimums: dict[str, dict[str, int]]
    band_minimums: dict[str, dict[str, int]]
    requested_rows: dict[str, int]
    real_repository_minimums: dict[str, int]
    real_repository_prompt_minimums: dict[str, int]
    real_repository_class_minimums: dict[str, dict[str, int]]


def _repository_class_minimums(
    value: int | Mapping[str, int | Mapping[str, int]],
    partition_names: list[str],
) -> dict[str, dict[str, int]]:
    """Expand real-repository class floors by partition and score class."""
    if isinstance(value, bool):
        raise ValueError("real-repository class minimums must be non-negative integers")
    if isinstance(value, int):
        raw = {partition: value for partition in partition_names}
    elif isinstance(value, Mapping):
        unknown = set(value) - set(partition_names)
        if unknown:
            raise ValueError(
                f"real-repository class minimums contain unknown partitions: {sorted(unknown)}"
            )
        raw = {partition: value.get(partition, 0) for partition in partition_names}
    else:
        raise ValueError("real-repository class minimums must be an integer or mapping")
    minimums = {}
    for partition, counts in raw.items():
        if isinstance(counts, bool):
            raise ValueError("real-repository class minimums must be non-negative integers")
        if isinstance(counts, int):
            values = {label: counts for label in _SCORE_CLASSES}
        elif isinstance(counts, Mapping):
            unknown = set(counts) - set(_SCORE_CLASSES)
            if unknown:
                raise ValueError(
                    f"real-repository class minimums contain unknown classes: {sorted(unknown)}"
                )
            values = {label: counts.get(label, 0) for label in _SCORE_CLASSES}
        else:
            raise ValueError("real-repository class minimums must be non-negative integers")
        if any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0
            for count in values.values()
        ):
            raise ValueError("real-repository class minimums must be non-negative integers")
        minimums[partition] = values
    return minimums


def validate_split_arguments(
    holdout_fractions: tuple[tuple[str, float], ...],
    min_real_repositories_per_split: int,
) -> list[str]:
    """Check the partition request itself, before any row is examined."""
    holdout_names = [name for name, _ in holdout_fractions]
    if (
        len(set(holdout_names)) != len(holdout_names)
        or "training" in holdout_names
    ):
        raise ValueError("holdout partition names must be unique and cannot be 'training'")
    total_holdout_fraction = sum(fraction for _, fraction in holdout_fractions)
    if any(
        not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or fraction <= 0
        for _, fraction in holdout_fractions
    ):
        raise ValueError("every holdout fraction must be positive")
    if total_holdout_fraction >= 1:
        raise ValueError("holdout fractions must sum to less than one")
    if (
        not isinstance(min_real_repositories_per_split, int)
        or isinstance(min_real_repositories_per_split, bool)
        or min_real_repositories_per_split < 0
    ):
        raise ValueError("real-repository minimum must be a non-negative integer")
    return holdout_names


def build_quotas(
    holdout_fractions: tuple[tuple[str, float], ...],
    *,
    min_real_repositories_per_split: int | Mapping[str, int],
    min_score_band_rows: int | Mapping[str, int],
    min_middle_score_band_rows: int | Mapping[str, int] | None,
    min_rows: int | Mapping[str, int],
    min_class_rows: int | Mapping[str, int],
    min_boundary_class_rows: int | Mapping[str, int],
    total_unique_prompts: int,
    min_real_repository_prompts: int | Mapping[str, int] = 0,
    min_real_repository_class_rows: int | Mapping[str, int | Mapping[str, int]] = 0,
) -> PartitionQuotas:
    """Resolve shared or partial floors into one comparable quota set."""
    if isinstance(min_real_repositories_per_split, Mapping):
        names = [name for name, _ in holdout_fractions] + ["training"]
        if set(min_real_repositories_per_split) - set(names):
            raise ValueError("real-repository minimums contain unknown partitions")
        real_repository_minimums = {
            name: min_real_repositories_per_split.get(name, 0) for name in names
        }
        holdout_names = validate_split_arguments(holdout_fractions, 0)
    else:
        holdout_names = validate_split_arguments(
            holdout_fractions, min_real_repositories_per_split
        )
        real_repository_minimums = {
            name: min_real_repositories_per_split
            for name in [*holdout_names, "training"]
        }
    partition_names = [*holdout_names, "training"]
    if any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0
        for count in real_repository_minimums.values()
    ):
        raise ValueError("real-repository minimums must be non-negative integers")

    real_repository_prompt_minimums = _minimums(
        min_real_repository_prompts,
        partition_names,
        name="real-repository prompt minimums",
        default=0,
    )
    real_repository_class_minimums = _repository_class_minimums(
        min_real_repository_class_rows, partition_names
    )
    common_band_minimums = _minimums(
        min_score_band_rows,
        partition_names,
        name="score-band minimums",
        default=1,
    )
    middle_band_minimums = (
        common_band_minimums
        if min_middle_score_band_rows is None
        else _minimums(
            min_middle_score_band_rows,
            partition_names,
            name="middle score-band minimums",
            default=0,
        )
    )
    band_minimums = {
        name: {
            band: (
                middle_band_minimums[name]
                if band == "40-59"
                else common_band_minimums[name]
            )
            for band in _SCORE_BANDS
        }
        for name in partition_names
    }
    row_minimums = _minimums(
        min_rows, partition_names, name="row minimums", default=0
    )
    boundary_minimums = _minimums(
        min_boundary_class_rows,
        partition_names,
        name="boundary score-band minimums",
        default=0,
    )
    per_class_minimums = _minimums(
        min_class_rows, partition_names, name="class minimums", default=0
    )
    class_minimums = {
        name: {
            **{label: max(1, per_class_minimums[name]) for label in _SCORE_CLASSES},
            **{label: boundary_minimums[name] for label in _PROXIMITY_CLASSES},
        }
        for name in partition_names
    }
    requested_rows = {
        name: max(row_minimums[name], round(total_unique_prompts * fraction))
        for name, fraction in holdout_fractions
    }
    requested_rows["training"] = row_minimums["training"]
    return PartitionQuotas(
        partition_names=partition_names,
        holdout_fractions=holdout_fractions,
        min_real_repositories_per_split=real_repository_minimums,
        row_minimums=row_minimums,
        class_minimums=class_minimums,
        band_minimums=band_minimums,
        requested_rows=requested_rows,
        real_repository_minimums=real_repository_minimums,
        real_repository_prompt_minimums=real_repository_prompt_minimums,
        real_repository_class_minimums=real_repository_class_minimums,
    )
