"""Validate data and output locations before loading training dependencies."""

import contracts as report
from data.validation import canonical_source_url

from .. import config, holdout, preflight
from ..training import artifacts

COMBINED_FILE = config.COMBINED_FILE
report_stale_artifacts = artifacts.report_stale_artifacts
verify_dataset = preflight.verify_dataset
verify_output_paths = preflight.verify_output_paths
require_test_coverage = holdout.require_test_coverage
load_combined_dataset = preflight.load_combined_dataset
separate_split_files = preflight.separate_split_files
_DEFAULT_INPUT_HOOKS = {
    "COMBINED_FILE": COMBINED_FILE,
    "report_stale_artifacts": report_stale_artifacts,
    "verify_dataset": verify_dataset,
    "verify_output_paths": verify_output_paths,
    "require_test_coverage": require_test_coverage,
    "load_combined_dataset": load_combined_dataset,
    "separate_split_files": separate_split_files,
}


def validate_inputs(
    train_file: str | None = None,
    validation_file: str | None = None,
    test_file: str | None = None,
    dataset_file: str | None = None,
    *,
    report_old_artifacts=None,
    combined_file: str | None = None,
    verify_rows=None,
    check_coverage=None,
    verify_paths=None,
    load_combined=None,
    load_separate=None,
    coverage_rows: int | None = None,
    resume_checkpoint: str | None = None,
    fork_checkpoint: str | None = None,
    release_only: bool = False,
    fresh_repository_test: bool = False,
):
    """Load, audit, and coverage-check selected partitions before model loading."""
    report_old_artifacts = report_old_artifacts or globals()["report_stale_artifacts"]
    combined_file = combined_file or COMBINED_FILE
    verify_rows = verify_rows or globals()["verify_dataset"]
    check_coverage = check_coverage or globals()["require_test_coverage"]
    verify_paths = verify_paths or globals()["verify_output_paths"]
    load_combined = load_combined or globals()["load_combined_dataset"]
    load_separate = load_separate or globals()["separate_split_files"]
    coverage_rows = coverage_rows or config.CHECK_ROWS
    report_old_artifacts()
    if dataset_file is None and not any((train_file, validation_file, test_file)):
        dataset_file = combined_file
    if dataset_file:
        if any((train_file, validation_file, test_file)):
            raise ValueError(
                "use either one combined dataset or three separate split files"
            )
        train_rows, validation_rows, test_rows = preflight.partition_or_exit(
            load_combined(dataset_file), dataset_file
        )
        selected_files = (dataset_file,)
    else:
        train_rows, validation_rows, test_rows, selected_files = load_separate(
            train_file, validation_file, test_file
        )
    families = [
        {row["source_type"] for row in split}
        for split in (train_rows, validation_rows, test_rows)
    ]
    print(
        "[i] source families by split: "
        f"train={sorted(families[0])}, validation={sorted(families[1])}, "
        f"test={sorted(families[2])}"
    )
    verify_rows(train_rows, validation_rows, test_rows, report)
    try:
        check_coverage(validation_rows, coverage_rows)
        if not fresh_repository_test:
            check_coverage(test_rows, coverage_rows)
        if fresh_repository_test:
            used_urls = {
                canonical_source_url(row["source_url"])
                for split in (train_rows, validation_rows)
                for row in split
                if row.get("source_type") == "real_repository"
            }
            test_rows = [
                row
                for row in test_rows
                if row.get("source_type") == "real_repository"
                and canonical_source_url(row["source_url"]) not in used_urls
            ]
            if not test_rows:
                raise SystemExit(
                    "the generated independent-test split has no fresh real-repository rows"
                )
            try:
                check_coverage(test_rows, coverage_rows)
            except SystemExit as exc:
                raise SystemExit(
                    f"fresh real-repository rows do not meet the independent-test minimums: {exc}"
                ) from None
            print(
                f"[i] independent test restricted to {len(test_rows)} fresh "
                "real-repository row(s); AIGCodeSet and synthetic rows are excluded."
            )
    except SystemExit as exc:
        raise SystemExit(f"[!] Holdout coverage gate failed: {exc}") from None
    protected = ("Modelfile", *selected_files)
    if resume_checkpoint is not None and fork_checkpoint is not None:
        raise ValueError("choose either an in-run resume or a fresh checkpoint fork")
    if fork_checkpoint is not None:
        from ..resume import validate_resume_checkpoint

        fork_checkpoint = validate_resume_checkpoint(fork_checkpoint)
        if not release_only:
            from ..resume_fork import verify_fork_outputs

            verify_fork_outputs(
                fork_checkpoint,
                config.RUN_DIR,
                config.ADAPTER_DIR,
                config.MERGE_DIR,
                protected,
                root=".",
            )
    elif resume_checkpoint is None and not release_only:
        verify_paths(
            config.RUN_DIR,
            config.ADAPTER_DIR,
            config.MERGE_DIR,
            root=".",
            protected_paths=protected,
        )
    elif resume_checkpoint is not None:
        from ..resume import validate_resume_checkpoint
        from ..resume_outputs import verify_resume_outputs
        from ..scoring.discovery import checkpoint_run_dir as get_checkpoint_run_dir

        resume_checkpoint = validate_resume_checkpoint(resume_checkpoint)
        checkpoint_run = get_checkpoint_run_dir(resume_checkpoint)
        verify_resume_outputs(
            checkpoint_run,
            config.ADAPTER_DIR,
            config.MERGE_DIR,
            protected,
            root=".",
        )
    return train_rows, validation_rows, test_rows
