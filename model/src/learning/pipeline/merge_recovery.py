"""Rebuild the merge and deployment of a selection whose release was interrupted.

A run claims its one-time independent test with a marker directory before scoring it, and
the gated release path saves the selected adapter only after that test passes. Losing power
between the adapter save and the merged-weight write therefore leaves every piece of
release evidence on disk while the ordinary ``train`` path refuses to continue, because
the claim is single-use and the reserve test must not be scored again.

This entry point rebuilds only the step that was lost. It refuses to proceed unless the
saved adapter postdates the claim, the selected checkpoint can be recovered from the run's
own validation diagnostics, and the saved weights still match that checkpoint tensor by
tensor. It then reuses the same gated merge and deployment a normal release uses, and
records the recovered provenance beside the merged weights. It never re-scores the reserve
test, never edits the claim marker, and never modifies saved checkpoints.

Configuration paths are resolved lazily: ``learning.config`` derives ``ADAPTER_DIR``,
``MERGE_DIR`` and ``GGUF`` from ``HACKGRADER_RUN_ID`` when it is first imported, so ``main``
pins that variable to the interrupted run before the model configuration loads. Pass
``settings`` and ``root`` to plan against another workspace instead, as the tests do.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Sequence
from typing import Any

from .recovery_outputs import (
    record_evidence,
    report_plan,
    report_planned_commands,
)
from .recovery_paths import RunPaths, validate_run_id
from .recovery_plan import Comparator, RecoveryPlan, RowCounter, plan_recovery


def _load_runtime() -> dict[str, Any]:
    """Import the GPU runtime only after the recovery checks have passed."""
    from ..runtime import load_dependencies

    return load_dependencies()


def _warn_gpu(torch: Any, minimum_vram: float) -> None:
    """Warn when another process left too little free VRAM for the merge."""
    from ..runtime import warn_if_gpu_busy

    warn_if_gpu_busy(torch, minimum_vram)


def _merge_and_deploy(plan: RecoveryPlan, dependencies: dict[str, Any]) -> None:
    """Run the gated merge and deployment with the recovered release evidence."""
    from . import models, release

    if os.path.realpath(release.MERGE_DIR) != os.path.realpath(plan.paths.merge_dir):
        raise SystemExit(
            "[!] The recovery plan and the release configuration name different merge "
            "directories; recovery stopped."
        )
    model, tokenizer = models.load_selected_model(dependencies, plan.paths.adapter_dir)
    release.merge_and_deploy(
        model,
        tokenizer,
        plan.training_rows,
        plan.adapter_file,
        validation_metrics=plan.validation_metrics,
        test_metrics=plan.claimed_test_metrics,
    )


def _deploy_merged(plan: RecoveryPlan) -> bool:
    """Convert and register an already merged build without loading the base model."""
    from ..deployment import run as deploy_model
    from ..deployment.provenance import MODELFILE

    return deploy_model(merge_dir=plan.paths.merge_dir, modelfile=MODELFILE)


def recover(
    *,
    run_id: str,
    dry_run: bool = False,
    root: str = ".",
    settings: RunPaths | None = None,
    load_runtime: Callable[[], dict[str, Any]] | None = None,
    merge_and_deploy: Callable[[RecoveryPlan, dict[str, Any]], None] | None = None,
    deploy: Callable[[RecoveryPlan], bool] | None = None,
    record: Callable[[RecoveryPlan], str] | None = None,
    warn_gpu: Callable[[Any, float], None] | None = None,
    digest: Callable[[str], str] | None = None,
    compare: Comparator | None = None,
    count_rows: RowCounter | None = None,
) -> int:
    """Rebuild the interrupted merge and deployment of one run's selected adapter."""
    plan = plan_recovery(
        run_id,
        root=root,
        settings=settings,
        digest=digest,
        compare=compare,
        count_rows=count_rows,
    )
    report_plan(plan)
    if dry_run:
        report_planned_commands(plan)
        return 0
    (record or record_evidence)(plan)
    if not plan.merge_required:
        if (deploy or _deploy_merged)(plan):
            return 0
        print("[!] The recovered build was not verified as registered with Ollama.")
        return 1
    load_runtime = load_runtime or _load_runtime
    dependencies = load_runtime()
    (warn_gpu or _warn_gpu)(dependencies["torch"], plan.paths.minimum_vram)
    (merge_and_deploy or _merge_and_deploy)(plan, dependencies)
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the recovery options for an interrupted release."""
    parser = argparse.ArgumentParser(
        prog="uv run src/cli.py recover",
        description=(
            "Rebuild the merge and deployment of an interrupted release whose one-time "
            "independent test was already claimed and passed."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="the interrupted run id, for example 20261008-193038",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="verify the recovered evidence and print the planned commands only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Pin the interrupted run id, then rebuild its merge and deployment."""
    options = _parser().parse_args(argv)
    run_id = validate_run_id(options.run_id)
    os.environ["HACKGRADER_RUN_ID"] = run_id
    if options.dry_run:
        print("[i] dry run: no model, merge, conversion, or registration will run.")
    return recover(run_id=run_id, dry_run=options.dry_run)
