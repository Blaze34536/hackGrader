"""Resolve which dataset files a training run reads, and say how to create them.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from typing import Any

import contracts as report

from ..config import COMBINED_FILE, TEST_FILE, TRAIN_FILE, VAL_FILE
from .rows import load_rows, partition_combined_rows


def candidate_path_for(path: str) -> str:
    """Suggest the matching candidate file for a missing training input."""
    if path.endswith(".candidate.jsonl"):
        return path
    return f"{path.removesuffix('.jsonl')}.candidate.jsonl"


def load_combined_dataset(path: str) -> list[dict[str, Any]]:
    """Load one JSONL and explain how to generate it when it is absent."""
    try:
        return load_rows(path, report)
    except FileNotFoundError:
        candidate_path = candidate_path_for(path)
        raise SystemExit(
            f"[!] Missing combined dataset {path!r}. Run "
            "./scripts/build_candidate_from_harvest.sh to harvest verified labels "
            f"and build one file with train/validation/test tags ({candidate_path!r})."
        ) from None


def partition_or_exit(
    rows: list[dict[str, Any]], path: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition a combined file and convert malformed split tags to a CLI error."""
    try:
        return partition_combined_rows(rows, path)
    except ValueError as exc:
        raise SystemExit(f"[!] Combined dataset split check failed: {exc}") from None


def separate_split_files(
    train_file: str | None = None,
    validation_file: str | None = None,
    test_file: str | None = None,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], tuple[str, ...]
]:
    """Load the three-file workflow, naming the candidate file when one is missing."""
    selected_files = (
        train_file or TRAIN_FILE,
        validation_file or VAL_FILE,
        test_file or TEST_FILE,
    )
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for name, path in zip(
        ("training", "validation", "independent test"), selected_files, strict=True
    ):
        try:
            rows_by_name[name] = load_rows(path, report)
        except FileNotFoundError:
            candidate_path = candidate_path_for(path)
            raise SystemExit(
                f"[!] Missing {name} data file {path!r}. Run the data command to "
                f"generate reviewable candidates (including {candidate_path!r}), "
                "then run train, or pass all three --*-file options."
            ) from None
    return (
        rows_by_name["training"],
        rows_by_name["validation"],
        rows_by_name["independent test"],
        selected_files,
    )


__all__ = [
    "COMBINED_FILE",
    "candidate_path_for",
    "load_combined_dataset",
    "partition_or_exit",
    "separate_split_files",
]
