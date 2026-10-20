"""Validate data and output paths before any training is allowed to start.

    from .preflight import load_rows, verify_dataset, verify_output_paths

Implementation modules:

``rows``     reads tagged JSONL and separates a combined file by its split column
``train_inputs``  resolves which files a run reads and how to create missing ones
``sizes``    requires enough distinct, class-balanced, boundary-supported prompts
``dataset``  requires three source-disjoint splits covering every source family
``paths``    refuses candidate and output paths that would overwrite something

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from .dataset import verify_dataset
from .paths import require_fresh_dir, resolve_candidate_paths, verify_output_paths
from .rows import DATASET_SPLITS, load_rows, partition_combined_rows
from .sizes import validate_dataset_sizes
from .train_inputs import (
    COMBINED_FILE,
    candidate_path_for,
    load_combined_dataset,
    partition_or_exit,
    separate_split_files,
)

__all__ = [
    "COMBINED_FILE",
    "DATASET_SPLITS",
    "candidate_path_for",
    "load_combined_dataset",
    "load_rows",
    "partition_combined_rows",
    "partition_or_exit",
    "require_fresh_dir",
    "resolve_candidate_paths",
    "separate_split_files",
    "validate_dataset_sizes",
    "verify_dataset",
    "verify_output_paths",
]
