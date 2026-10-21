"""Command-line entry point for model data, training, and deployment tasks.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    """Build a command parser while leaving subcommand options to each workflow."""
    parser = argparse.ArgumentParser(description="HackGrader model workspace commands.")
    parser.add_argument(
        "command",
        choices=("data", "harvest", "train", "deploy", "score", "recover"),
        help=(
            "build data, harvest an audited repository cohort, train or preflight, "
            "inspect checkpoints, deploy a model, or rebuild an interrupted release"
        ),
    )
    return parser


def _train_parser() -> argparse.ArgumentParser:
    """Build training options for selecting one combined dataset or separate files."""
    parser = argparse.ArgumentParser(
        prog="uv run src/cli.py train",
        description=(
            "Validate datasets, train, and deploy only when the selected checkpoint "
            "passes validation and final-test release gates."
        ),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help="combined JSONL with train, validation, and test rows",
    )
    parser.add_argument(
        "-c",
        "--candidates",
        action="store_true",
        help="use model/data/dataset.candidate.jsonl as the combined dataset",
    )
    parser.add_argument("--train-file", help="training JSONL file (use all three split-file options)")
    parser.add_argument(
        "--validation-file",
        "--val-file",
        dest="validation_file",
        help="validation JSONL file (use all three split-file options)",
    )
    parser.add_argument("--test-file", help="independent-test JSONL file (use all three split-file options)")
    parser.add_argument(
        "--resume-checkpoint",
        help="continue inside the checkpoint's original run and remaining time cap",
    )
    parser.add_argument(
        "--fork-checkpoint",
        help="start a fresh run and time cap from a saved adapter; preserve its source run",
    )
    parser.add_argument(
        "--release-only",
        action="store_true",
        help="validate, test, and release a forked adapter without starting training",
    )
    parser.add_argument(
        "--fresh-repositories",
        help="build a new audited dataset from this repository list and train a checkpoint fork",
    )
    parser.add_argument(
        "--strict-gates",
        action="store_true",
        help="disable provisional release and stop when adaptive validation tuning bottoms out",
    )
    parser.add_argument(
        "--fresh-repository-test",
        action="store_true",
        help="use only final-test repository URLs absent from train and validation",
    )
    parser.add_argument(
        "-p",
        "--preflight-only",
        action="store_true",
        help="validate inputs and output paths, then exit before model loading or training",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to candidate generation, fine-tuning, or model deployment."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments[:1])
    remaining = arguments[1:]
    shared_source = str(MODEL_ROOT.parent / "src")
    if shared_source not in sys.path:
        sys.path.insert(0, shared_source)
    os.chdir(MODEL_ROOT)
    if args.command == "data":
        from data.builder import main as run_data_builder

        run_data_builder(remaining)
        return 0
    if args.command == "harvest":
        from data.harvest import main as run_harvest

        return run_harvest(remaining)
    if args.command == "train":
        train_parser = _train_parser()
        options = train_parser.parse_args(remaining)
        explicit_files = (
            options.train_file,
            options.validation_file,
            options.test_file,
        )
        if options.candidates and (
            options.dataset or options.fresh_repositories or any(explicit_files)
        ):
            train_parser.error(
                "--candidates cannot be combined with --dataset or split files"
            )
        if options.resume_checkpoint and options.fork_checkpoint:
            train_parser.error(
                "--resume-checkpoint and --fork-checkpoint are mutually exclusive"
            )
        if options.release_only and not options.fork_checkpoint:
            train_parser.error("--release-only requires --fork-checkpoint")
        if options.release_only and options.preflight_only:
            train_parser.error("--release-only cannot be combined with --preflight-only")
        if options.fresh_repositories and (
            options.dataset or any(explicit_files) or options.release_only
        ):
            train_parser.error(
                "--fresh-repositories builds its own combined data and cannot be "
                "combined with explicit data files or --release-only"
            )
        if options.fresh_repositories and not options.fork_checkpoint:
            train_parser.error(
                "--fresh-repositories requires --fork-checkpoint so training starts "
                "from the existing model in a protected new run"
            )
        if options.dataset and any(explicit_files):
            train_parser.error("--dataset cannot be combined with separate split files")
        if any(explicit_files) and not all(explicit_files):
            train_parser.error(
                "provide all three of --train-file, --validation-file, and --test-file"
            )

        from learning import distributed

        if not options.preflight_only and not options.release_only:
            if options.fresh_repositories:
                print(
                    "[i] --fresh-repositories builds its data in this process, so this "
                    "invocation trains on one GPU; rerun `train --dataset <built file>` "
                    "to train on every GPU."
                )
            else:
                command = distributed.worker_command(arguments, script=__file__)
                if command is not None:
                    return distributed.launch(command)

        from learning.pipeline import main as run_training

        if options.fresh_repositories:
            import time
            import uuid

            from data.builder import main as build_candidate

            options.dataset = (
                f"dataset.recovery-{time.strftime('%Y%m%d-%H%M%S')}-"
                f"{uuid.uuid4().hex[:8]}.candidate.jsonl"
            )
            build_candidate(
                [
                    "--repos",
                    options.fresh_repositories,
                    "--combined-out",
                    options.dataset,
                    "--fresh-repository-test",
                ]
            )
            options.strict_gates = True
            options.fresh_repository_test = True
        if options.candidates:
            from data.config import DEFAULT_CANDIDATE

            options.dataset = DEFAULT_CANDIDATE
        run_training(
            fresh_repository_test=options.fresh_repository_test,
            strict_gates=options.strict_gates,
            train_file=explicit_files[0],
            validation_file=explicit_files[1],
            test_file=explicit_files[2],
            dataset_file=options.dataset,
            preflight_only=options.preflight_only,
            resume_checkpoint=options.resume_checkpoint,
            fork_checkpoint=options.fork_checkpoint,
            release_only=options.release_only,
        )
        return 0
    if args.command == "score":
        from learning.scoring import main as run_scoring

        return run_scoring(remaining)
    if args.command == "recover":
        from learning.pipeline.merge_recovery import main as recover_release

        return recover_release(remaining)
    from learning.deployment import main as run_deployment

    return run_deployment(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
