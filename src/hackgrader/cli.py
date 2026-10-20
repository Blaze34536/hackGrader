"""Command-line interface for grading projects in a Devpost gallery.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import argparse

from .pipeline.workflow import run_gallery


def _parser() -> argparse.ArgumentParser:
    """Build the application command parser and describe its single gallery argument."""
    parser = argparse.ArgumentParser(description="Grade projects from a Devpost gallery.")
    parser.add_argument(
        "devpost_url",
        help="URL of the Devpost hackathon gallery whose projects should be graded",
    )
    return parser


def main() -> None:
    """Read a gallery URL from argv and execute the grading workflow."""
    args = _parser().parse_args()
    run_gallery(args.devpost_url)


if __name__ == "__main__":
    main()
