"""Require three source-disjoint splits that together cover every source family.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from typing import Any

from data.config import MIN_REAL_REPOSITORIES_PER_SPLIT
from data.validation import REQUIRED_SOURCE_TYPES, canonical_source_url
from data.verification import verify_splits

from .sizes import validate_dataset_sizes


def verify_dataset(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    report_contract: Any,
) -> None:
    """Require three source-disjoint splits with coverage and score diversity."""
    splits = (
        ("training", train_rows),
        ("validation", validation_rows),
        ("independent test", test_rows),
    )
    try:
        validate_dataset_sizes(train_rows, validation_rows, test_rows, report_contract)
        for name, rows in splits:
            real_repositories = {
                canonical_source_url(row["source_url"])
                for row in rows
                if row["source_type"] == "real_repository"
            }
            if len(real_repositories) < MIN_REAL_REPOSITORIES_PER_SPLIT:
                raise ValueError(
                    f"{name} contains {len(real_repositories)} independent real "
                    f"repositories; at least {MIN_REAL_REPOSITORIES_PER_SPLIT} are "
                    "required"
                )
        verify_splits(list(splits), report_contract)
    except ValueError as exc:
        raise SystemExit(f"[!] Training data gate failed: {exc}") from None

    all_rows = train_rows + validation_rows + test_rows
    source_groups = {group for row in all_rows for group in row["source_groups"]}
    print(
        f"[i] provenance: {len(source_groups)} distinct source groups; "
        f"{sum(row['source_type'] == 'real_repository' for row in train_rows)} "
        "real-repository training view(s), "
        f"{sum(row['source_type'] == 'real_repository' for row in validation_rows)} "
        "validation view(s), "
        f"{sum(row['source_type'] == 'real_repository' for row in test_rows)} "
        "test view(s)"
    )
    if set(REQUIRED_SOURCE_TYPES) != {row["source_type"] for row in all_rows}:
        raise SystemExit(
            "[!] Data splits do not cover all required source families: "
            f"{sorted(REQUIRED_SOURCE_TYPES)}. A snippet-only build "
            "(HACKGRADER_SNIPPET_ONLY=1) omits real_repository by design."
        )
