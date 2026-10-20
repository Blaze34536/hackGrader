"""Write and validate score-only exports and inspect saved validation cutoffs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..holdout import summarize_holdout
from .measure import sweep_thresholds


def export_predictions(
    destination: str,
    predictions: list[dict[str, Any]],
    *,
    step: int,
    split: str,
    run_dir: str,
    prevalence: float | None,
    threshold: int,
    extra_metadata: dict[str, Any] | None = None,
    include_generated_output: bool = False,
) -> str:
    """Atomically create one prediction JSONL in the model workspace, never overwrite."""
    model_root = Path(__file__).resolve().parents[3].resolve()
    requested = Path(destination)
    if not requested.is_absolute():
        requested = model_root / requested
    parent = requested.parent.resolve()
    try:
        if os.path.commonpath((str(model_root), str(parent))) != str(model_root):
            raise SystemExit(
                "[!] Prediction exports must stay inside the model directory."
            )
    except ValueError:
        raise SystemExit(
            "[!] Prediction exports must stay inside the model directory."
        ) from None
    if requested.suffix.lower() != ".jsonl":
        raise SystemExit("[!] Prediction export path must end in .jsonl.")
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / requested.name
    if output.exists() or output.is_symlink():
        raise SystemExit(f"[!] Refusing to overwrite prediction export {output}.")
    metadata = _export_metadata(step, split, run_dir, prevalence, threshold, model_root)
    metadata.update(extra_metadata or {})
    exported_predictions = []
    for index, original in enumerate(predictions, start=1):
        record = dict(original)
        record["repository_id"] = (
            record.get("source_url") or record.get("prompt_sha256") or f"row:{index}"
        )
        if not include_generated_output:
            record.pop("generated_output", None)
        exported_predictions.append(record)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=parent
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps({"metadata": metadata}, ensure_ascii=False) + "\n")
            for record in exported_predictions:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if output.exists() or output.is_symlink():
            raise SystemExit(f"[!] Refusing to overwrite prediction export {output}.")
        os.link(temporary, output)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    else:
        os.remove(temporary)
    print(f"[i] wrote {len(exported_predictions)} prediction row(s) to {output}.")
    return str(output)


def _export_metadata(step, split, run_dir, prevalence, threshold, model_root):
    """Build portable metadata without exposing repository prompt text."""
    resolved = Path(run_dir).resolve()
    try:
        recorded_run_dir = str(resolved.relative_to(model_root))
    except ValueError:
        recorded_run_dir = str(resolved)
    metadata = {
        "checkpoint": step,
        "run_dir": recorded_run_dir,
        "split": split,
        "prevalence": prevalence,
        "export_threshold": threshold,
    }
    return metadata


def load_prediction_export(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate export metadata and per-example score records."""
    try:
        with open(path, encoding="utf-8") as handle:
            header = json.loads(next(handle))
            predictions = [json.loads(line) for line in handle if line.strip()]
    except (OSError, StopIteration, json.JSONDecodeError):
        raise SystemExit(
            "[!] Prediction export could not be read as valid JSONL."
        ) from None
    metadata = header.get("metadata") if isinstance(header, dict) else None
    if (
        not isinstance(metadata, dict)
        or metadata.get("split") != "validation"
        or not isinstance(metadata.get("checkpoint"), int)
    ):
        raise SystemExit(
            "[!] Threshold calibration accepts validation prediction exports only."
        )
    prevalence = metadata.get("prevalence")
    if prevalence is not None and (
        not isinstance(prevalence, (int, float)) or not 0.0 < prevalence < 1.0
    ):
        raise SystemExit("[!] Prediction export prevalence is invalid.")
    _validate_prediction_rows(predictions)
    return metadata, predictions


def _validate_prediction_rows(predictions: list[dict[str, Any]]) -> None:
    """Reject malformed prediction records and exports without score rows."""
    for record in predictions:
        if not isinstance(record, dict) or not isinstance(record.get("parsed"), bool):
            raise SystemExit("[!] Prediction export contains an invalid row record.")
        if record["parsed"] and any(
            not isinstance(record.get(key), int) or not 0 <= record[key] <= 100
            for key in ("target_score", "predicted_score")
        ):
            raise SystemExit(
                "[!] Prediction export contains scores outside 0 through 100."
            )
    if not predictions:
        raise SystemExit("[!] Prediction export has no example records.")


def sweep_export(path: str, thresholds: list[int]) -> int:
    """Print gate results for cutoffs using saved validation predictions only."""
    metadata, predictions = load_prediction_export(path)
    try:
        records = sweep_thresholds(
            predictions,
            thresholds,
            split="Validation holdout",
            prevalence=metadata.get("prevalence"),
        )
    except ValueError as exc:
        raise SystemExit(f"[!] Could not sweep thresholds: {exc}") from None
    _print_threshold_records(records)
    print(
        f"[i] predictions: checkpoint-{metadata['checkpoint']}; {len(predictions)} selected validation rows"
    )
    print(
        "[!] Thresholds are diagnostics only: the exported validation set was used to "
        "choose among them. This does not authorize deployment or change the production cutoff."
    )
    return 0


def _print_threshold_records(records) -> None:
    """Render threshold rows and their metric summaries."""
    header = f"{'cutoff':>6} {'prec':>6} {'recall':>7} {'spec':>6} {'sep':>7} {'MAE':>6}  gates"
    print(header)
    print("-" * len(header))
    for threshold, metrics in records:
        failed = ",".join(metrics["failed_gates"]) or "pass"
        print(
            f"{threshold:>6} {metrics['precision']:>6.2f} {metrics['recall']:>7.2f} "
            f"{metrics['specificity']:>6.2f} {metrics['class_separation']:>7.1f} "
            f"{metrics['mean_absolute_error']:>6.1f}  {failed}"
        )
        print(summarize_holdout(metrics))
