"""Report held-out metrics for saved checkpoints without saving or deploying anything.

Training spent its whole budget before it evaluated anything, so a model that missed a
release gate left no deployable artifact behind even though every checkpoint was on disk.
This package scores checkpoints that already exist, and it is also what training uses to
choose which checkpoint to deploy instead of trusting a validation loss that is dominated
by memorising the report template.

    from .scoring import best_checkpoint, evaluate_split, report_sweep

Implementation modules:

``discovery``  finds run directories and saved checkpoints, and loads the holdouts
``measure``    scores one checkpoint, then ranks checkpoints against each other
``report``     renders the score table and the sweep's recommendation
``sweep``      scores every checkpoint by swapping adapters on a resident model
``cli``        the read-only ``score`` subcommand

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from .cli import main
from .discovery import (
    adapter_directory,
    checkpoint_directory,
    checkpoint_run_dir,
    checkpoint_steps,
    load_scoring_splits,
    newest_run_dir,
    outputs_root,
)
from .measure import (
    SPLIT_LABELS,
    TEST_LABEL,
    VALIDATION_LABEL,
    best_checkpoint,
    evaluate_split,
    rank_key,
    score_checkpoint,
    sweep_thresholds,
)
from .report import format_score_table, pass_by_step, report_sweep
from .sweep import sweep_loaded_model

__all__ = [
    "SPLIT_LABELS",
    "TEST_LABEL",
    "VALIDATION_LABEL",
    "adapter_directory",
    "best_checkpoint",
    "checkpoint_directory",
    "checkpoint_run_dir",
    "checkpoint_steps",
    "evaluate_split",
    "format_score_table",
    "load_scoring_splits",
    "main",
    "newest_run_dir",
    "outputs_root",
    "pass_by_step",
    "rank_key",
    "report_sweep",
    "score_checkpoint",
    "sweep_loaded_model",
    "sweep_thresholds",
]
