"""Coordinate safe preflight, runtime budgets, and model-release phases."""

from __future__ import annotations

import time
from typing import Any

from .. import config, distributed, runtime
from ..resume_runtime import active_input_paths
from ..training import artifacts
from . import final_test, release
from .release_only import release_forked_checkpoint
from .resume import initialize_run, prepare_checkpoint_run
from .training import run_training


def _setting(options, hooks, name, default):
    """Resolve one optional pipeline hook or CLI override."""
    return options.get(name) or hooks.get(name) or default


def main(*, hooks: dict[str, Any], **options) -> None:
    """Run training or a no-training release of a forked saved checkpoint."""
    release_only = options.get("release_only", False)
    strict_gates = options.get("strict_gates", False)
    fresh_repository_test = options.get("fresh_repository_test", False)
    validate = _setting(options, hooks, "validate_inputs", hooks["_validate_inputs"])
    estimate = _setting(
        options, hooks, "estimate_training", hooks["_estimate_training"]
    )
    load_runtime = _setting(
        options, hooks, "load_dependencies", runtime.load_dependencies
    )
    warn_gpu = _setting(options, hooks, "warn_if_gpu_busy", runtime.warn_if_gpu_busy)
    make_model = _setting(options, hooks, "prepare_model", runtime.prepare_model)
    report = _setting(options, hooks, "report_budget", hooks["_report_budget"])
    resume_checkpoint = options.get("resume_checkpoint") or config.RESUME_CHECKPOINT
    fork_checkpoint = options.get("fork_checkpoint")
    if release_only and not fork_checkpoint:
        raise SystemExit("[!] Release-only mode requires --fork-checkpoint.")
    if release_only and options.get("preflight_only", False):
        raise SystemExit("[!] Use either --preflight-only or --release-only, not both.")
    if resume_checkpoint is not None and fork_checkpoint is not None:
        raise SystemExit("[!] Choose either in-run resume or a fresh checkpoint fork.")
    started_at = time.monotonic()
    run_hours = hooks.get("HOURS", config.HOURS)
    reserves = hooks.get(
        "VALIDATION_SCORING_RESERVE_HOURS", config.VALIDATION_SCORING_RESERVE_HOURS
    ) + hooks.get("FINAL_TEST_RESERVE_HOURS", config.FINAL_TEST_RESERVE_HOURS)
    dataset = options.get("dataset_file")
    train_file = options.get("train_file")
    validation_file = options.get("validation_file")
    test_file = options.get("test_file")
    active = active_input_paths(dataset, train_file, validation_file, test_file)
    run_dir = hooks.get("RUN_DIR", config.RUN_DIR)
    train_rows, validation_rows, test_rows, continuation, fork_checkpoint, run_dir = (
        prepare_checkpoint_run(
            validate,
            train_file=train_file,
            validation_file=validation_file,
            test_file=test_file,
            dataset_file=dataset,
            resume_checkpoint=resume_checkpoint,
            fork_checkpoint=fork_checkpoint,
            run_dir=run_dir,
            release_only=release_only,
            fresh_repository_test=fresh_repository_test,
        )
    )
    distributed.rendezvous(what="the preflight check", timeout=3600.0)
    if distributed.is_main():
        print(f"[i] This run writes checkpoints to {run_dir}")
    if options.get("preflight_only", False):
        estimate(len(train_rows) * hooks.get("REPEAT", config.REPEAT))
        print("[i] Preflight passed; no model loaded and no training started.")
        return
    if release_only:
        deadline = started_at + run_hours * 3600
        return release_forked_checkpoint(
            load_runtime,
            warn_gpu,
            report,
            fork_checkpoint,
            validation_rows,
            test_rows,
            len(train_rows) * hooks.get("REPEAT", config.REPEAT),
            deadline
            - hooks.get("FINAL_TEST_RESERVE_HOURS", config.FINAL_TEST_RESERVE_HOURS)
            * 3600,
            deadline,
            minimum_vram=hooks.get("MIN_FREE_VRAM_GB", config.MIN_FREE_VRAM_GB),
            save_adapter=options.get("save_adapter")
            or hooks.get("_save_adapter")
            or hooks.get("save_adapter")
            or release.save_adapter,
            merge_and_deploy=_setting(
                options, hooks, "merge_and_deploy", release.merge_and_deploy
            ),
            blocked=_setting(options, hooks, "blocked", release.blocked),
            remaining=_setting(
                options,
                hooks,
                "require_time_remaining",
                artifacts.require_time_remaining,
            ),
            require_real_repository_test=fresh_repository_test,
            train_rows=train_rows,
        )
    estimate(len(train_rows) * hooks.get("REPEAT", config.REPEAT))
    initialize = (
        options.get("initialize_run_directory")
        or hooks.get("initialize_run_directory")
        or artifacts.initialize_run_directory
    )
    deadline, remaining_seconds = initialize_run(
        initialize,
        run_dir=run_dir,
        resume_checkpoint=resume_checkpoint,
        fork_checkpoint=fork_checkpoint,
        active=active,
        run_hours=run_hours,
        reserves=reserves,
        started_at=started_at,
    )
    validation_deadline = (
        deadline
        - hooks.get("FINAL_TEST_RESERVE_HOURS", config.FINAL_TEST_RESERVE_HOURS) * 3600
    )
    minimum_vram = hooks.get("MIN_FREE_VRAM_GB", config.MIN_FREE_VRAM_GB)
    save = (
        options.get("save_adapter")
        or hooks.get("_save_adapter")
        or hooks.get("save_adapter")
        or release.save_adapter
    )
    deploy = _setting(options, hooks, "merge_and_deploy", release.merge_and_deploy)
    blocked = _setting(options, hooks, "blocked", release.blocked)
    remaining = _setting(
        options, hooks, "require_time_remaining", artifacts.require_time_remaining
    )
    run_training(
        hooks=hooks,
        options=options,
        load_runtime=load_runtime,
        warn_gpu=warn_gpu,
        make_model=make_model,
        prepare=hooks["_prepare_datasets"],
        report=report,
        run_rounds=_setting(options, hooks, "run_training_rounds", None),
        final=_setting(
            options,
            hooks,
            "final_test",
            hooks.get("_final_test") or final_test.run_final_test,
        ),
        save=save,
        deploy=deploy,
        block=blocked,
        remaining=remaining,
        estimate=estimate,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        fresh_repository_test=fresh_repository_test,
        continuation_checkpoint=continuation,
        resume_checkpoint=resume_checkpoint,
        run_dir=run_dir,
        strict_gates=strict_gates,
        started_at=started_at,
        run_hours=run_hours,
        remaining_seconds=remaining_seconds,
        training_deadline=deadline - reserves * 3600,
        validation_deadline=validation_deadline,
        deadline=deadline,
        minimum_vram=minimum_vram,
        repeat=hooks.get("REPEAT", config.REPEAT),
    )
