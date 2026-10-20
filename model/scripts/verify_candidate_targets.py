"""Check that a candidate dataset's repository targets are measured, not banded.

A candidate is only worth training on when its real-repository rows carry the percentage the
harvest attributed and appear in every split. This reads the candidate back and reports that
distribution; it exits non-zero when the rows are missing, land in one split only, or carry a
single repeated number — which is what a band constant looks like and what the served model
reproduced.

Usage, from the model workspace:

    uv run --project . python scripts/verify_candidate_targets.py

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import model_paths

model_paths.add_shared_source()

import contracts as report  # noqa: E402
from data.config import DEFAULT_CANDIDATE  # noqa: E402

CANDIDATE = DEFAULT_CANDIDATE
BANDS = ((0, 39), (40, 59), (60, 100))


def band_of(score: int) -> str:
    """Return the named score band a target belongs to."""
    for low, high in BANDS:
        if low <= score <= high:
            return f"{low}-{high}"
    return "outside"


def main() -> int:
    """Report the candidate's target distribution and fail when it cannot support training."""
    path = Path(CANDIDATE)
    if not path.exists():
        print(f"[!] {path} does not exist; nothing to verify.")
        return 1
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    repositories = [row for row in rows if row.get("source_type") == "real_repository"]
    splits = collections.Counter(row.get("split") for row in rows)
    print(f"[i] {len(rows)} row(s): " + ", ".join(f"{k} {v}" for k, v in sorted(splits.items())))

    if not repositories:
        print("[!] No real-repository rows: the candidate cannot train the repository head.")
        return 1

    by_split: dict[str, list[int]] = collections.defaultdict(list)
    for row in repositories:
        parsed = report.parse_report(row["messages"][-1]["content"])
        if parsed["ai_score"] is not None:
            by_split[str(row.get("split"))].append(parsed["ai_score"])
    scores = [score for values in by_split.values() for score in values]
    print(
        f"[i] {len(repositories)} repository row(s): mean {statistics.mean(scores):.1f}, "
        f"median {statistics.median(scores):.0f}, range {min(scores)}-{max(scores)}"
    )
    bands = collections.Counter(band_of(score) for score in scores)
    print("[i] score bands: " + ", ".join(f"{name} {count}" for name, count in sorted(bands.items())))
    for split, values in sorted(by_split.items()):
        print(f"[i] {split}: {len(values)} row(s), range {min(values)}-{max(values)}")

    failures = []
    if statistics.pstdev(scores) < 1.0:
        failures.append(
            "repository targets are constant, which is the band-midpoint behavior the "
            "verified label path exists to replace"
        )
    for split in ("train", "validation", "test"):
        if not by_split.get(split):
            failures.append(f"no real-repository rows in the {split} split")
    for message in failures:
        print(f"[!] {message}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
