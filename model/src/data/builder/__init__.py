"""Generate one tagged JSONL candidate with source-disjoint data partitions.

    from data.builder import main

Implementation modules:

``candidates``  builds every candidate row and validates it against the data contract
``splits``      quarantines conflicts, deduplicates, caps, splits, and re-verifies
``cli``         parses options, guards output paths, and publishes the candidate file

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from .candidates import build_candidate_rows, real_repositories, repository_manifest
from .cli import _output_paths, _parser, main
from .splits import split_candidate_rows, split_rows

__all__ = [
    "_output_paths",
    "_parser",
    "build_candidate_rows",
    "main",
    "real_repositories",
    "repository_manifest",
    "split_candidate_rows",
    "split_rows",
]
