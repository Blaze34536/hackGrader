"""Validate outputs for a fresh-budget run forked from a saved adapter."""

from . import preflight
from .resume import validate_resume_checkpoint
from .scoring.discovery import checkpoint_run_dir


def verify_fork_outputs(
    checkpoint: str,
    run_dir: str,
    adapter_dir: str,
    merge_dir: str,
    active_paths: tuple[str, ...],
    *,
    root: str = ".",
) -> str:
    """Validate the source checkpoint and fresh outputs without touching the old run."""
    checkpoint = validate_resume_checkpoint(checkpoint)
    source_run = checkpoint_run_dir(checkpoint)
    preflight.verify_output_paths(
        run_dir,
        adapter_dir,
        merge_dir,
        root=root,
        protected_paths=(*active_paths, source_run),
    )
    return checkpoint
