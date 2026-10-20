"""Resolve the run, adapter, merge, and dataset paths a recovery may touch.

The learning package keeps sources free of hash comments and under 200 lines, so the
recovery path lives in several small modules: this one resolves paths and digests, its
neighbour recovers artifact evidence, the next assembles a plan, another renders and
records it, and the entry module rebuilds the release.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

RUN_ID_PATTERN = re.compile(r"^\d{8}-\d{6}$")
ATTEMPT_MARKER = ".independent-test-attempted"
RECOVERY_RECORD = "merge_recovery.json"
STAMP_NAME = "merge_stamp.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"
DIAGNOSTICS_PREFIX = "validation-checkpoint-"
TENSOR_TOLERANCE = 2e-2


@dataclass(frozen=True)
class RunPaths:
    """Configured locations for one run, resolved before any file is touched."""

    run_id: str
    run_dir: str
    adapter_dir: str
    merge_dir: str
    gguf: str
    tag: str
    dataset: str
    base_model: str
    max_len: int
    repeat: int
    minimum_vram: float


def validate_run_id(run_id: str) -> str:
    """Require a bare run id so every derived artifact path stays inside the workspace."""
    if not RUN_ID_PATTERN.fullmatch(run_id or ""):
        raise SystemExit(
            f"[!] {run_id!r} is not a run id of the form YYYYMMDD-HHMMSS; recovery stopped."
        )
    return run_id


def configured_paths(run_id: str) -> RunPaths:
    """Resolve configured artifact paths, requiring the pinned run id to match."""
    from .. import config
    from ..deployment.provenance import TAG
    from ..scoring.discovery import outputs_root

    if run_id != config.RUN_ID:
        raise SystemExit(
            "[!] The run id must be pinned before the model configuration is imported; "
            f"recovery saw {config.RUN_ID!r} instead of {run_id!r}."
        )
    return RunPaths(
        run_id=run_id,
        run_dir=os.path.join(outputs_root(), run_id),
        adapter_dir=config.ADAPTER_DIR,
        merge_dir=config.MERGE_DIR,
        gguf=config.GGUF,
        tag=TAG,
        dataset=config.COMBINED_FILE,
        base_model=config.BASE_MODEL,
        max_len=config.MAX_LEN,
        repeat=config.REPEAT,
        minimum_vram=config.MIN_FREE_VRAM_GB,
    )


def resolve_paths(paths: RunPaths, root: str) -> RunPaths:
    """Resolve the workspace-relative paths of one run against the workspace root."""

    def resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(root, path)

    return replace(
        paths,
        run_dir=resolve(paths.run_dir),
        adapter_dir=resolve(paths.adapter_dir),
        merge_dir=resolve(paths.merge_dir),
        dataset=resolve(paths.dataset),
    )


def run_started_at(run_dir: str) -> float:
    """Read the run's original start time from its metadata."""
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            started = float(json.load(handle)["started_at"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise SystemExit(
            f"[!] {metadata_path} has no usable start time; recovery stopped."
        ) from None
    return started


def adapter_digest(path: str) -> str:
    """Return the digest the release stamp and the versioned GGUF name derive from."""
    from ..deployment.provenance import sha256_file

    return sha256_file(Path(path))
