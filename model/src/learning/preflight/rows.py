"""Read training examples and separate a combined file by its split tags.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
import os
from typing import Any

from data.config import DATA_SPLITS
from data.validation import validate_row

DATASET_SPLITS = DATA_SPLITS


def load_rows(
    path: str, report_contract: Any, *, repeat: int = 1
) -> list[dict[str, Any]]:
    """Read training examples and reject any row that violates the report contract."""
    if repeat < 1:
        raise ValueError("training-row repeat count must be positive")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                raise TypeError(f"{path} row {index} is not a message-based example")
            try:
                validate_row(row, path, index, report_contract)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path} row {index} failed validation: {exc}"
                ) from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no training examples")
    print(f"[i] {path}: {len(rows)} row(s)")
    return rows * repeat


def partition_combined_rows(
    rows: list[dict[str, Any]], path: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate one JSONL's rows by required training, validation, and test tags."""
    partitions: dict[str, list[dict[str, Any]]] = {
        split: [] for split in DATASET_SPLITS
    }
    for index, row in enumerate(rows):
        split = row.get("split")
        if not isinstance(split, str) or split not in partitions:
            raise ValueError(
                f"{path} row {index} has invalid split tag {split!r}; expected one of "
                f"{', '.join(DATASET_SPLITS)}"
            )
        partitions[split].append(
            {key: value for key, value in row.items() if key != "split"}
        )
    missing = [split for split, split_rows in partitions.items() if not split_rows]
    if not rows:
        missing = list(DATASET_SPLITS)
    if missing:
        raise ValueError(
            f"{path} is missing rows for split(s): {', '.join(missing)}; "
            "generate one combined JSONL containing all three splits"
        )
    return tuple(partitions[split] for split in DATASET_SPLITS)
