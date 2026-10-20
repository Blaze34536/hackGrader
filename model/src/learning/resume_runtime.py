"""Resolve an existing run and enforce its original wall-clock budget."""

import os

from . import config
from .resume import remaining_training_seconds, validate_resume_checkpoint
from .scoring.discovery import checkpoint_run_dir


def active_input_paths(
    dataset: str | None, train: str | None, validation: str | None, test: str | None
) -> tuple[str, ...]:
    """List protected data/config paths for the chosen combined or split layout."""
    if dataset:
        return "Modelfile", dataset
    return (
        "Modelfile",
        train or config.TRAIN_FILE,
        validation or config.VAL_FILE,
        test or config.TEST_FILE,
    )


def resolve_resume_run(checkpoint: str, configured_run: str) -> tuple[str, str]:
    """Return the validated checkpoint and its run, honoring explicit path overrides."""
    checkpoint = validate_resume_checkpoint(checkpoint)
    run_dir = checkpoint_run_dir(checkpoint)
    configured = os.environ.get("HACKGRADER_OUTPUT_DIR")
    if configured and os.path.realpath(configured) != os.path.realpath(run_dir):
        raise SystemExit("[!] HACKGRADER_OUTPUT_DIR must match the checkpoint's run.")
    return checkpoint, run_dir or configured_run


def remaining_resume_budget(run_dir: str, hours: float, reserves: float) -> float:
    """Require enough original run time remains for resumed training and release checks."""
    remaining = remaining_training_seconds(run_dir, hours)
    if remaining <= reserves * 3600:
        raise SystemExit(
            "[!] The original training cap has no time left after validation/test "
            "reserves; checkpoints remain untouched and restart stopped."
        )
    return remaining
