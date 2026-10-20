"""Validate score command options and coordinate a read-only checkpoint sweep."""

from __future__ import annotations

import argparse
from typing import Any

from ..config import CHECK_ROWS
from ..holdout import AI_THRESHOLD, summarize_holdout
from ..runtime import load_dependencies
from .diagnostics import save_validation_diagnostics
from .discovery import checkpoint_steps, load_scoring_splits, newest_run_dir
from .exports import export_predictions, sweep_export
from .measure import SPLIT_LABELS, score_checkpoint
from .report import report_sweep


def parser() -> argparse.ArgumentParser:
    """Build command-line options for checkpoint scoring and validation calibration."""
    command = argparse.ArgumentParser(
        prog="uv run src/cli.py score",
        description="Report held-out metrics without saving weights or registering a model.",
    )
    command.add_argument("-r", "--run-dir", help="run directory holding checkpoints")
    command.add_argument(
        "-k",
        "--checkpoint-step",
        type=int,
        action="append",
        default=None,
        help="score this saved step; may be provided more than once",
    )
    command.add_argument(
        "-a", "--all-checkpoints", action="store_true", help="score every checkpoint in the run"
    )
    command.add_argument(
        "-s", "--split", choices=("validation", "test", "both"), default="validation"
    )
    command.add_argument(
        "-n", "--limit", type=int, default=CHECK_ROWS, help="validation prompts to score"
    )
    command.add_argument(
        "--threshold", type=int, default=AI_THRESHOLD, help="classification cutoff (0-100)"
    )
    command.add_argument("--batch-size", type=int, default=1, help="inference batch size")
    command.add_argument(
        "--prediction-file", help="saved validation predictions for cutoff calibration"
    )
    command.add_argument(
        "--sweep-thresholds", help="comma-separated cutoffs for saved-prediction calibration"
    )
    command.add_argument(
        "--export-predictions", help="write validation predictions to this JSONL file"
    )
    command.add_argument("-d", "--dataset", help="combined candidate JSONL to score")
    command.add_argument("--validation-file", help="validation JSONL (with --test-file)")
    command.add_argument("--test-file", help="test JSONL (diagnostic scoring is validation-only)")
    return command


def chosen_checkpoints(available, steps, every):
    """Select requested saved checkpoint steps or default to the newest adapter."""
    if every:
        return available
    if steps:
        wanted = set(steps)
        chosen = [(step, path) for step, path in available if step in wanted]
        missing = sorted(wanted - {step for step, _path in chosen})
        if missing:
            raise SystemExit(f"[!] No saved checkpoint for step(s): {missing}")
        return chosen
    return available[-1:]


def parse_thresholds(raw: str) -> list[int]:
    """Parse a stable, unique list of integer score cutoffs."""
    try:
        thresholds = sorted({int(value.strip()) for value in raw.split(",")})
    except ValueError:
        raise SystemExit(
            "[!] --sweep-thresholds must be comma-separated integers."
        ) from None
    if not thresholds or any(not 0 <= value <= 100 for value in thresholds):
        raise SystemExit("[!] Swept thresholds must be between zero and one hundred.")
    return thresholds


def main(argv: list[str] | None = None) -> int:
    """Parse scoring options, sweep requested checkpoints, and print a report."""
    args = parser().parse_args(argv)
    if _calibration_requested(args):
        return sweep_export(
            args.prediction_file, parse_thresholds(args.sweep_thresholds)
        )
    _validate_score_options(args)
    run_dir = args.run_dir or newest_run_dir()
    available = checkpoint_steps(run_dir)
    if not available:
        raise SystemExit(f"[!] No checkpoint-* adapter directories under {run_dir!r}.")
    chosen = chosen_checkpoints(available, args.checkpoint_step, args.all_checkpoints)
    if args.export_predictions and len(chosen) != 1:
        raise SystemExit("[!] Export one checkpoint at a time with --checkpoint-step.")
    validation, test = load_scoring_splits(
        dataset_file=args.dataset,
        validation_file=args.validation_file,
        test_file=args.test_file,
    )
    names = ("validation", "test") if args.split == "both" else (args.split,)
    splits = {
        name: (validation if name == "validation" else test, SPLIT_LABELS[name])
        for name in names
    }
    return _score_checkpoints(args, run_dir, chosen, names, splits)


def _calibration_requested(args) -> bool:
    """Validate the exclusive saved-prediction calibration command shape."""
    if not args.prediction_file and not args.sweep_thresholds:
        return False
    if not args.prediction_file or not args.sweep_thresholds:
        raise SystemExit(
            "[!] --prediction-file and --sweep-thresholds must be used together."
        )
    if args.split != "validation" or args.export_predictions:
        raise SystemExit(
            "[!] Cutoff calibration is validation-only and cannot export again."
        )
    if args.all_checkpoints or args.checkpoint_step or args.run_dir:
        raise SystemExit(
            "[!] Cutoff sweeps use only --prediction-file, not checkpoint options."
        )
    return True


def _validate_score_options(args) -> None:
    """Reject invalid run, batch, cutoff, and export options before GPU load."""
    if args.split in {"test", "both"}:
        raise SystemExit(
            "[!] Independent-test scoring is disabled in the diagnostic command; "
            "the test may be used once only by the gated release workflow."
        )
    if args.all_checkpoints and args.checkpoint_step:
        raise SystemExit(
            "[!] Use either --all-checkpoints or --checkpoint-step, not both."
        )
    if args.limit < 1 or args.batch_size < 1:
        raise SystemExit("[!] --limit and --batch-size must be positive.")
    if not 0 <= args.threshold <= 100:
        raise SystemExit("[!] --threshold must be between zero and one hundred.")
    if args.export_predictions and args.split != "validation":
        raise SystemExit(
            "[!] Only validation predictions may be exported for calibration."
        )


def _score_checkpoints(args, run_dir, chosen, names, splits) -> int:
    """Load dependencies once, score adapters, and print the comparison."""
    print(
        f"[i] scoring {len(chosen)} checkpoint(s) from {run_dir} at {args.limit} row(s) per split: {', '.join(names)}"
    )
    print(
        "[i] read-only: this command saves no adapter, merged checkpoint, GGUF, or model tag."
    )
    dependencies = load_dependencies()
    records: list[tuple[int, str, dict[str, Any]]] = []
    for step, path in chosen:
        print(f"[i] checkpoint-{step}: loading adapter from {path}")
        results = score_checkpoint(
            dependencies,
            path,
            splits,
            args.limit,
            classification_threshold=args.threshold,
            batch_size=args.batch_size,
        )
        for name, metrics in results.items():
            records.append((step, SPLIT_LABELS[name], metrics))
            if name == "validation":
                save_validation_diagnostics(run_dir, step, path, metrics)
            print(summarize_holdout(metrics))
            if args.export_predictions:
                export_predictions(
                    args.export_predictions,
                    metrics["predictions"],
                    step=step,
                    split=name,
                    run_dir=run_dir,
                    prevalence=metrics.get("prevalence"),
                    threshold=args.threshold,
                )
    report_sweep(records)
    return 0
