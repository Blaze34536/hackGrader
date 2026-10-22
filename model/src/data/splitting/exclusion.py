"""Keep held-out benchmark repositories out of every partition.

The human-bias benchmark (``scripts/measure_human_bias.py``) asks what the judge says about
repositories no agent wrote. A repository that also trained the judge measures fit, not
calibration, and the 2026-10-18 cohort took all twenty of its human labels from that pool.
Dropping the listed repositories before splitting keeps the benchmark independent of every
partition, test included.

Author: Nihal M
Date: 2026
License: MIT
"""

from __future__ import annotations

from typing import Any

from ..validation import canonical_source_url


def read_repository_list(path: str) -> set[str]:
    """Return the canonical URLs a list file names, ignoring blank lines and comments."""
    try:
        with open(path, encoding="utf-8") as handle:
            lines = [line.split("#", 1)[0].strip() for line in handle]
    except OSError as exc:
        raise SystemExit(f"[!] Repository list {path} could not be read: {exc}") from None
    return {canonical_source_url(line) for line in lines if line}


def exclude_repositories(rows: list[dict[str, Any]], path: str | None) -> list[dict[str, Any]]:
    """Return the rows whose source repository is not on the list at ``path``."""
    if not path:
        return rows
    excluded = read_repository_list(path)
    kept = [
        row for row in rows if canonical_source_url(row.get("source_url", "")) not in excluded
    ]
    if len(kept) != len(rows):
        print(
            f"[i] Held out {len(rows) - len(kept)} row(s) from benchmark repositories "
            f"listed in {path}; they appear in no partition."
        )
    return kept


__all__ = ["exclude_repositories", "read_repository_list"]
