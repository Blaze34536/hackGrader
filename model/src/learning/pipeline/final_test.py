"""Run the one-time independent test after validation or a reached training cap."""

import os
from typing import Any

from data.validation import canonical_source_url

from .. import config, holdout, scoring
from ..scoring.discovery import checkpoint_run_dir
from ..training.artifacts import require_time_remaining
from . import models, release


def _claim_independent_test(selection) -> None:
    """Atomically reserve the run's sole independent-test attempt before inference."""
    run_dir = checkpoint_run_dir(selection.checkpoint_path)
    if not os.path.isfile(os.path.join(run_dir, "run_metadata.json")):
        raise SystemExit(
            "[!] Run metadata is missing; independent-test scoring cannot be safely "
            "recorded, so evaluation stopped."
        )
    marker = os.path.join(run_dir, ".independent-test-attempted")
    try:
        os.mkdir(marker)
    except FileExistsError:
        raise SystemExit(
            "[!] The independent test has already been attempted for this run."
        ) from None
    selection.test_attempted = True


def run_final_test(
    dependencies: dict[str, Any],
    selection,
    test_rows: list[dict[str, Any]],
    deadline: float,
    *,
    load_model=None,
    score=None,
    report_gates=None,
    summarize=None,
    block_release=None,
    allow_provisional_validation: bool = False,
    require_real_repository_test: bool = False,
    train_rows: list[dict[str, Any]] | None = None,
    validation_rows: list[dict[str, Any]] | None = None,
):
    """Score validation's selection on the independent test exactly once."""
    load_model = load_model or models.load_selected_model
    score = score or scoring.evaluate_split
    report_gates = report_gates or holdout.report_release_gates
    summarize = summarize or holdout.summarize_holdout
    block_release = block_release or release.blocked
    validation_passed = bool(selection.metrics["passed"])
    if require_real_repository_test:
        if train_rows is None or validation_rows is None:
            raise SystemExit(
                "[!] Fresh repository test requires explicit training and validation "
                "provenance for source-disjointness; evaluation stopped."
            )
        try:
            repositories = {
                canonical_source_url(row["source_url"])
                for row in test_rows
                if row.get("source_type") == "real_repository"
            }
            used_repositories = {
                canonical_source_url(row["source_url"])
                for row in train_rows + validation_rows
                if row.get("source_type") == "real_repository"
            }
        except (KeyError, TypeError, AttributeError):
            raise SystemExit(
                "[!] Fresh independent-test provenance is incomplete; evaluation stopped."
            ) from None
        if (
            not test_rows
            or len(repositories) < 2
            or repositories & used_repositories
            or len(repositories) != len(test_rows)
        ):
            raise SystemExit(
                "[!] The fresh independent test needs only source-disjoint "
                "real-repository examples from at least two repositories; "
                "evaluation stopped."
            )
    if getattr(selection, "test_attempted", False):
        raise SystemExit(
            "[!] The independent test has already been attempted for this selection."
        )
    if allow_provisional_validation and (
        not selection.cap_reached or validation_passed
    ):
        raise SystemExit(
            "[!] Provisional release requires a failed validation gate at the training cap."
        )
    if not validation_passed and not allow_provisional_validation:
        raise SystemExit(
            "[!] Validation did not pass; independent-test scoring and deployment stopped."
        )
    if not validation_passed:
        print(
            "[!] Training cap reached with failed validation gates; scoring the "
            "independent test once to decide whether a provisional tag is allowed."
        )
    require_time_remaining(deadline, "final independent test")
    model, tokenizer = load_model(dependencies, selection.checkpoint_path)
    require_time_remaining(deadline, "final independent test")
    print(
        f"[i] final validation candidate: checkpoint-{selection.active} "
        f"({selection.checkpoint_path})."
    )
    if not validation_passed:
        for failure in holdout.format_gate_failures(selection.metrics):
            print(f"[!] Provisional validation failure: {failure}")
    else:
        report_gates(scoring.VALIDATION_LABEL, selection.metrics)
    print(
        "[i] Gate metrics compare against generated teaching labels; they are not "
        "human-verified authorship results."
    )
    _claim_independent_test(selection)
    test_metrics = score(
        model,
        tokenizer,
        test_rows,
        config.CHECK_ROWS,
        split=scoring.TEST_LABEL,
        deadline=deadline,
    )
    print(summarize(test_metrics))
    report_gates(scoring.TEST_LABEL, test_metrics)
    if not test_metrics["passed"]:
        if test_metrics["violations"]:
            block_release(
                selection.metrics,
                test_metrics,
                checkpoint_path=selection.checkpoint_path,
            )
        raise SystemExit(
            "[!] Independent-test evaluation did not pass its release gate; "
            "deployment stopped."
        )
    selection.test_metrics = test_metrics
    selection.provisional = not validation_passed
    if selection.provisional:
        selection.metrics["release_status"] = "provisional"
        selection.metrics["cap_reached"] = True
        selection.metrics["independent_test_passed"] = True
    if not validation_passed:
        print(
            "[!] Independent test passed, but validation failed; this checkpoint is "
            "eligible only for a separate provisional tag, not the normal release tag."
        )
    print(
        f"[i] final test: precision {test_metrics['precision']:.2f}, "
        f"recall {test_metrics['recall']:.2f}, specificity "
        f"{test_metrics['specificity']:.2f}, balanced accuracy "
        f"{test_metrics['balanced_accuracy']:.2f}"
    )
    return model, tokenizer
