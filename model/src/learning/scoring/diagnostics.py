"""Persist validation-only gate diagnostics beside a training run."""

from pathlib import Path
from typing import Any

from .exports import export_predictions


def save_validation_diagnostics(run_dir, step, checkpoint_path, metrics):
    """Save diagnostics when predictions exist; warn rather than fail selection."""
    if not isinstance(metrics.get("predictions"), list):
        return None
    try:
        path = export_validation_diagnostics(
            run_dir, step, checkpoint_path, metrics
        )
    except (OSError, SystemExit, ValueError) as exc:
        print(f"[!] Could not save validation diagnostics: {exc}")
        return None
    print(f"[i] validation diagnostics saved to {path}")
    return path


def export_validation_diagnostics(
    run_dir: str,
    step: int,
    checkpoint_path: str,
    metrics: dict[str, Any],
) -> str:
    """Store failed-gate evidence and generated responses without source prompts."""
    predictions = metrics.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("validation diagnostics require row-level predictions")
    base = Path(run_dir) / f"validation-checkpoint-{step}.jsonl"
    destination = base
    attempt = 2
    while destination.exists() or destination.is_symlink():
        destination = base.with_name(f"{base.stem}-{attempt:02d}{base.suffix}")
        attempt += 1
    fields = (
        "passed", "failed_gates", "examples", "unparsed", "unparsed_rate",
        "ai_support", "human_support", "mean_absolute_error", "score_spread",
        "class_separation", "precision", "recall", "specificity", "f1",
        "balanced_accuracy", "true_positive", "false_positive", "true_negative",
        "false_negative", "format_failures", "violations", "uncertainty",
    )
    return export_predictions(
        str(destination),
        predictions,
        step=step,
        split="validation",
        run_dir=run_dir,
        prevalence=metrics.get("prevalence"),
        threshold=metrics.get("classification_threshold", 40),
        extra_metadata={
            "checkpoint_path": checkpoint_path,
            "gate_metrics": {key: metrics.get(key) for key in fields},
        },
        include_generated_output=True,
    )
