"""Run one candidate generation: parse options, build rows, split, and publish.

Nothing here overwrites an active training dataset. Outputs must use ``*.candidate.jsonl``
filenames; the default candidate can be replaced atomically after the complete new data
has passed all checks.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import argparse
import os
import random

from learning.preflight import resolve_candidate_paths

from ..config import (
    AIG_ROWS,
    BENCHMARK_REPOS_FILE,
    DEFAULT_CANDIDATE,
    DEFAULT_TEST_CANDIDATE,
    DEFAULT_TRAIN_CANDIDATE,
    DEFAULT_VAL_CANDIDATE,
    REPO_REPEAT,
    TARGET_CAP,
)
from ..publishing import report_fit, write_combined_jsonl, write_jsonl
from ..rows import label_histogram, report_mix, report_separation
from .candidates import build_candidate_rows
from .splits import split_rows

_ACTIVE_PATHS = (
    "data/dataset.jsonl",
    "data/dataset.val.jsonl",
    "data/dataset.test.jsonl",
)


def _parser() -> argparse.ArgumentParser:
    """Create command-line options for one candidate generation run."""
    parser = argparse.ArgumentParser(description="Build the HackGrader dataset.")
    parser.add_argument("-n", "--rows", type=int, default=AIG_ROWS, help="AIGCodeSet rows per label")
    parser.add_argument("-r", "--repos", default=None, help="file of real repository URLs")
    parser.add_argument(
        "--repository-labels",
        default=None,
        help=(
            "JSONL manifest of fail-closed verified repository authorship labels; "
            "repositories without an accepted label are excluded"
        ),
    )
    parser.add_argument(
        "--exclude-repositories",
        default=BENCHMARK_REPOS_FILE,
        help=(
            "file of repository URLs held out of every partition, so the human-bias "
            f"benchmark stays independent; '' keeps them (default: {BENCHMARK_REPOS_FILE})"
        ),
    )
    parser.add_argument(
        "--fresh-repository-test",
        action="store_true",
        help="reserve a two-class real-repository holdout for fresh-test preflight",
    )
    parser.add_argument(
        "--min-fresh-test-repository-prompts",
        type=int,
        default=50,
        help="minimum distinct real-repository prompts in the fresh test (default: 50)",
    )
    parser.add_argument(
        "--min-fresh-test-class-rows",
        type=int,
        default=10,
        help="minimum prompts per score class in the fresh test (default: 10)",
    )
    parser.add_argument(
        "--repo-repeat",
        type=int,
        default=REPO_REPEAT,
        help="repeat each real-repository training row this many times",
    )
    parser.add_argument(
        "--target-cap",
        type=int,
        default=TARGET_CAP,
        help="maximum candidate rows retained for any one target score",
    )
    parser.add_argument("-s", "--seed", type=int, default=13, help="random seed for sampling and splitting")
    parser.add_argument(
        "-o",
        "--combined-out",
        default=DEFAULT_CANDIDATE,
        help=(
            "write all three source-disjoint splits to one tagged JSONL "
            f"(default: {DEFAULT_CANDIDATE}; replaces this file only after a successful build)"
        ),
    )
    parser.add_argument(
        "--separate-files",
        action="store_true",
        help="write three candidate files instead of one combined JSONL",
    )
    parser.add_argument(
        "--train-out",
        default=DEFAULT_TRAIN_CANDIDATE,
        help=(
            "training candidate output when --separate-files is used "
            f"(default: {DEFAULT_TRAIN_CANDIDATE})"
        ),
    )
    parser.add_argument(
        "--val-out",
        default=DEFAULT_VAL_CANDIDATE,
        help=(
            "validation candidate output when --separate-files is used "
            f"(default: {DEFAULT_VAL_CANDIDATE})"
        ),
    )
    parser.add_argument(
        "--test-out",
        default=DEFAULT_TEST_CANDIDATE,
        help=(
            "independent test candidate output when --separate-files is used "
            f"(default: {DEFAULT_TEST_CANDIDATE})"
        ),
    )
    return parser


def _output_paths(args) -> tuple[str, ...]:
    """Resolve and guard candidate outputs, permitting only the default replacement."""
    if args.separate_files:
        if args.combined_out != DEFAULT_CANDIDATE:
            raise SystemExit("[!] --combined-out cannot be combined with --separate-files.")
        return resolve_candidate_paths(
            args.train_out,
            args.val_out,
            args.test_out,
            active_paths=_ACTIVE_PATHS,
            root=".",
        )
    if any(
        path != default
        for path, default in (
            (args.train_out, DEFAULT_TRAIN_CANDIDATE),
            (args.val_out, DEFAULT_VAL_CANDIDATE),
            (args.test_out, DEFAULT_TEST_CANDIDATE),
        )
    ):
        raise SystemExit("[!] Use --separate-files when specifying split-specific outputs.")
    return resolve_candidate_paths(
        args.combined_out,
        active_paths=_ACTIVE_PATHS,
        root=".",
        allow_existing=(DEFAULT_CANDIDATE,),
    )


def _write_separate(
    output_paths: tuple[str, ...],
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
) -> None:
    """Write three new candidate files, removing partial output if one write fails."""
    written: list[str] = []
    try:
        for path, partition in zip(
            output_paths, (train_rows, val_rows, test_rows), strict=True
        ):
            write_jsonl(path, partition)
            written.append(path)
    except BaseException:
        for candidate in written:
            if os.path.exists(candidate):
                os.remove(candidate)
        raise
    print(
        f"Wrote {len(train_rows)} training, {len(val_rows)} validation, and "
        f"{len(test_rows)} final-test candidate row(s) to separate files."
    )


def _write_combined(path: str, partitions: tuple[list[dict], list[dict], list[dict]]) -> int:
    """Write a complete combined candidate, replacing the default only after success."""
    output_path = os.path.abspath(path)
    default_path = os.path.abspath(DEFAULT_CANDIDATE)
    replace_default = os.path.normcase(os.path.realpath(output_path)) == os.path.normcase(
        os.path.realpath(default_path)
    )
    return write_combined_jsonl(path, partitions, replace=replace_default)


def main(argv: list[str] | None = None) -> None:
    """Generate an audited combined candidate without replacing active datasets."""
    args = _parser().parse_args(argv)
    output_paths = _output_paths(args)
    rng = random.Random(args.seed)
    rows, weights = build_candidate_rows(args)
    train_rows, val_rows, test_rows = split_rows(rows, args, rng)

    spread = label_histogram(train_rows)
    report_mix(
        train_rows,
        sum(row["source_kind"] == "repository" for row in train_rows),
        spread,
    )
    report_separation(train_rows, weights)
    report_fit(train_rows, val_rows + test_rows)

    if args.separate_files:
        _write_separate(output_paths, train_rows, val_rows, test_rows)
    else:
        total_rows = _write_combined(
            output_paths[0], (train_rows, val_rows, test_rows)
        )
        print(
            f"Wrote {total_rows} tagged candidate row(s) to {output_paths[0]} "
            f"(training={len(train_rows)}, validation={len(val_rows)}, "
            f"independent_test={len(test_rows)})."
        )
    print("[i] Active training, validation, and test datasets were left untouched.")
