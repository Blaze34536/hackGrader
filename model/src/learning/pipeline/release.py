"""Retain failed checkpoints and deploy only a fully verified passing model."""

import json
import os
from typing import Any

from learning.deployment import MODELFILE
from learning.deployment import run as deploy_model

from ..config import ADAPTER_DIR, BASE_MODEL, MAX_LEN, MERGE_DIR, RUN_DIR
from ..training.artifacts import require_time_remaining, verify_merge, write_stamp

PROVISIONAL_TAG = "hackgrader-provisional"


def save_adapter(model, tokenizer=None) -> str:
    """Persist the selected adapter before any merge or deployment operation."""
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    adapter_file = os.path.join(ADAPTER_DIR, "adapter_model.safetensors")
    if not os.path.isfile(adapter_file):
        raise SystemExit(
            f"[!] {ADAPTER_DIR} has no adapter_model.safetensors, so the selected "
            "adapter was not written and deployment stopped."
        )
    print(f"[i] selected adapter written to {ADAPTER_DIR}.")
    return adapter_file


def merge_and_deploy(
    model,
    tokenizer,
    row_count: int,
    adapter_file: str,
    *,
    deadline: float | None = None,
    provisional: bool = False,
    validation_metrics: dict[str, Any] | None = None,
    test_metrics: dict[str, Any] | None = None,
) -> None:
    """Merge verified weights and deploy the matching model tag before the deadline."""
    if not provisional and (
        validation_metrics is None
        or not validation_metrics.get("passed")
        or test_metrics is None
        or test_metrics.get("split") != "Final independent test"
        or not test_metrics.get("passed")
    ):
        raise SystemExit(
            "[!] Normal release requires passing validation and independent-test "
            "metrics; no merge or registration was performed."
        )
    if provisional and (
        validation_metrics is None
        or validation_metrics.get("passed")
        or validation_metrics.get("cap_reached") is not True
        or test_metrics is None
        or test_metrics.get("split") != "Final independent test"
        or not test_metrics.get("passed")
    ):
        raise SystemExit(
            "[!] Provisional deployment requires failed validation at the training cap "
            "and a passing independent-test result; no merge or registration was performed."
        )
    require_time_remaining(deadline, "final merge")
    os.makedirs(os.path.dirname(MERGE_DIR) or ".", exist_ok=True)
    model.save_pretrained_merged(MERGE_DIR, tokenizer, save_method="merged_16bit")
    tokenizer.save_pretrained(MERGE_DIR)
    verify_merge(MERGE_DIR, adapter_file)
    stamp = write_stamp(MERGE_DIR, adapter_file, row_count, BASE_MODEL, MAX_LEN)
    stamp["release_status"] = "provisional" if provisional else "passed"
    stamp["validation_metrics"] = _release_metric_summary(validation_metrics or {})
    stamp["independent_test_metrics"] = _release_metric_summary(test_metrics or {})
    from learning.deployment import STAMP as stamp_name

    with open(os.path.join(MERGE_DIR, stamp_name), "w", encoding="utf-8") as handle:
        json.dump(stamp, handle, indent=2)
    print(f"[i] merged adapter {stamp['adapter_sha256'][:12]} from {row_count} row(s).")
    require_time_remaining(deadline, "deployment")
    if not deploy_model(
        merge_dir=MERGE_DIR,
        modelfile=MODELFILE,
        tag=PROVISIONAL_TAG if provisional else None,
    ):
        tag = PROVISIONAL_TAG if provisional else "configured release tag"
        raise SystemExit(
            f"[!] Model was not verified as deployed to {tag}; pipeline remains unchanged."
        )
    if provisional:
        print(
            f"[!] Provisional model registered as {PROVISIONAL_TAG}; it was not "
            "promoted to the normal default tag."
        )


def _release_metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Persist compact gate evidence without storing holdout prompts or predictions."""
    fields = (
        "split",
        "passed",
        "cap_reached",
        "independent_test_passed",
        "release_status",
        "failed_gates",
        "examples",
        "unparsed",
        "unparsed_rate",
        "mean_absolute_error",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "format_failures",
    )
    return {name: metrics.get(name) for name in fields}


def save_and_deploy(
    model,
    tokenizer,
    row_count: int,
    *,
    provisional: bool = False,
    validation_metrics: dict[str, Any] | None = None,
    test_metrics: dict[str, Any] | None = None,
) -> None:
    """Write the verified adapter and merged weights, then register its chosen tag."""
    adapter_file = save_adapter(model, tokenizer)
    merge_and_deploy(
        model,
        tokenizer,
        row_count,
        adapter_file,
        provisional=provisional,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
    )


def blocked(
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None = None,
    *,
    checkpoint_path: str | None = None,
) -> None:
    """Explain failed release gates without merging or deploying any model."""
    failed = [
        f"{metrics['split']}: {', '.join(metrics['failed_gates'])}"
        for metrics in (validation_metrics, test_metrics)
        if metrics is not None and metrics["failed_gates"]
    ]
    if test_metrics is None:
        next_step = (
            "Review validation diagnostics and improve training data, labels, or settings. "
            "Do not tune on the reserved independent test or lower gates."
        )
    elif not validation_metrics.get("passed", False) and test_metrics.get(
        "passed", False
    ):
        next_step = (
            "A provisional checkpoint may be registered only under the separate "
            f"{PROVISIONAL_TAG} tag; it is not a normal release."
        )
    else:
        next_step = (
            "The independent test did not pass. Do not tune against this test; use "
            "validation evidence and reserve a new independent test next run."
        )
    details = [
        "[!] The release gates failed; no checkpoint was deployed.",
        f"    Failed gates: {'; '.join(failed)}.",
        "    Nothing was merged or registered; the existing Ollama model and "
        "HackGrader configuration are unchanged.",
    ]
    if test_metrics is None:
        details.append(
            "    Final independent test was not scored because validation did not pass."
        )
    details.append(f"    Saved checkpoints retained at: {RUN_DIR}")
    if checkpoint_path:
        details.append(f"    Best validation checkpoint retained at: {checkpoint_path}")
    if os.path.isfile(os.path.join(ADAPTER_DIR, "adapter_model.safetensors")):
        details.append(f"    Selected adapter retained at: {ADAPTER_DIR}")
    details.append(f"    Next step: {next_step}")
    raise SystemExit("\n".join(details))
