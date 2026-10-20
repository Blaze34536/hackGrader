"""Connect the public pipeline entry point to its canonical components."""

from .. import config, holdout, preflight, runtime, scoring
from .. import inputs as data_inputs
from ..training import artifacts, loop
from . import final_test, inputs, models, release, runner


def _validate_inputs(
    train_file: str | None = None,
    validation_file: str | None = None,
    test_file: str | None = None,
    dataset_file: str | None = None,
    resume_checkpoint: str | None = None,
    fork_checkpoint: str | None = None,
    release_only: bool = False,
    fresh_repository_test: bool = False,
):
    """Validate selected rows, holdout coverage, and outputs before model loading."""
    return inputs.validate_inputs(
        train_file,
        validation_file,
        test_file,
        dataset_file,
        resume_checkpoint=resume_checkpoint,
        fork_checkpoint=fork_checkpoint,
        release_only=release_only,
        fresh_repository_test=fresh_repository_test,
    )


def _prepare_datasets(tokenizer, train_rows, validation_rows):
    """Prepare separate validation and training conversations."""
    eval_rows = data_inputs.stratified(validation_rows, config.CHECK_ROWS)
    eval_dataset = data_inputs.build_dataset(tokenizer, eval_rows, what="validation")
    repeated_rows = train_rows * config.REPEAT
    train_dataset = data_inputs.build_dataset(
        tokenizer, repeated_rows, shuffle=True, what="train"
    )
    print(f"[i] target AI scores: {data_inputs.label_spread(repeated_rows)}")
    return train_dataset, eval_dataset


def _report_budget() -> None:
    """Print the hard cap and time protected for validation and final testing."""
    print(
        f"[i] validation-guided continuation has a {config.HOURS:.1f}-hour cap; "
        f"reserves are {config.VALIDATION_SCORING_RESERVE_HOURS:.1f}h for validation and "
        f"{config.FINAL_TEST_RESERVE_HOURS:.1f}h for one final-test pass."
    )


def _final_test(
    dependencies,
    selected,
    test_rows,
    deadline,
    *,
    allow_provisional_validation: bool = False,
    require_real_repository_test: bool = False,
    train_rows=None,
    validation_rows=None,
):
    """Score the selected validation result once on the independent test."""
    return final_test.run_final_test(
        dependencies,
        selected,
        test_rows,
        deadline,
        load_model=lambda deps, path: models.load_selected_model(deps, path),
        score=scoring.evaluate_split,
        report_gates=holdout.report_release_gates,
        summarize=holdout.summarize_holdout,
        block_release=release.blocked,
        allow_provisional_validation=allow_provisional_validation,
        require_real_repository_test=require_real_repository_test,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )


def main(**options) -> None:
    """Run the capped training workflow or validation-only adapter release."""
    hooks = {
        "_estimate_training": artifacts.check_training_budget,
        "_final_test": _final_test,
        "_prepare_datasets": _prepare_datasets,
        "_report_budget": _report_budget,
        "_round_model": models.load_round_model,
        "_save_adapter": release.save_adapter,
        "_validate_inputs": _validate_inputs,
        "blocked": release.blocked,
        "build_dataset": data_inputs.build_dataset,
        "evaluate_split": scoring.evaluate_split,
        "initialize_run_directory": artifacts.initialize_run_directory,
        "label_spread": data_inputs.label_spread,
        "load_dependencies": runtime.load_dependencies,
        "merge_and_deploy": release.merge_and_deploy,
        "prepare_model": runtime.prepare_model,
        "report_release_gates": holdout.report_release_gates,
        "report_stale_artifacts": artifacts.report_stale_artifacts,
        "require_test_coverage": holdout.require_test_coverage,
        "run_training_rounds": loop.run_training_rounds,
        "save_adapter": release.save_adapter,
        "separate_split_files": preflight.separate_split_files,
        "stratified": data_inputs.stratified,
        "summarize_holdout": holdout.summarize_holdout,
        "verify_dataset": preflight.verify_dataset,
        "verify_output_paths": preflight.verify_output_paths,
        "warn_if_gpu_busy": runtime.warn_if_gpu_busy,
        "ADAPTER_DIR": config.ADAPTER_DIR,
        "CHECK_ROWS": config.CHECK_ROWS,
        "COMBINED_FILE": config.COMBINED_FILE,
        "FINAL_TEST_RESERVE_HOURS": config.FINAL_TEST_RESERVE_HOURS,
        "HOURS": config.HOURS,
        "MERGE_DIR": config.MERGE_DIR,
        "MIN_FREE_VRAM_GB": config.MIN_FREE_VRAM_GB,
        "REPEAT": config.REPEAT,
        "RUN_DIR": config.RUN_DIR,
        "VALIDATION_SCORING_RESERVE_HOURS": config.VALIDATION_SCORING_RESERVE_HOURS,
        "TEST_LABEL": scoring.TEST_LABEL,
        "VALIDATION_LABEL": scoring.VALIDATION_LABEL,
    }
    return runner.main(hooks=hooks, **options)


__all__ = ["main"]
