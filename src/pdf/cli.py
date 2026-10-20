"""Render scores.json into printable HackGrader PDF reports.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import argparse
import json
from pathlib import Path

from grading.reporting import Reporting

from .context import build_context, cover_context
from .renderer import count_pages, overlong_lines, write_pdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "scores.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "report.pdf"


def main(argv: list[str] | None = None) -> int:
    """Render a scores JSON file into a PDF and report layout diagnostics."""
    parser = argparse.ArgumentParser(description="Render annotated scores as a PDF report.")
    parser.add_argument(
        "input", nargs="?", type=Path, default=DEFAULT_INPUT, help="scores JSON file"
    )
    parser.add_argument(
        "output", nargs="?", type=Path, default=DEFAULT_OUTPUT, help="output PDF path"
    )
    args = parser.parse_args(argv)
    input_path = args.input
    output_path = args.output
    if not input_path.exists():
        print(f"Error: {input_path} was not found. Run the grading pipeline first.")
        return 1
    with open(input_path, encoding="utf-8") as file:
        scores = json.load(file)
    scores = Reporting().annotate_all(scores)
    projects = build_context(scores)
    cover = cover_context(input_path, projects)
    errors = write_pdf(output_path, projects, cover)
    if errors:
        print("Error: PDF generation failed; the renderer reported the problem above.")
        return 1
    overlong = overlong_lines(projects)
    if overlong:
        print(f"[!] {overlong} line(s) still cross the margin; check the wrap constants.")
    pages = count_pages(output_path)
    expected = len(projects) + 1  # the cover page
    if pages is not None:
        if pages < expected:
            print(
                f"[!] only {pages} page(s) for {len(projects)} project(s) and a cover "
                "page; every project should open its own page, so check that page "
                "breaks are applied."
            )
        elif pages > expected:
            # More pages than projects is the normal outcome: each project opens a new page
            # and its evidence and deductions flow onto the next one.
            print(
                f"[i] {pages} page(s) for {len(projects)} project(s) and a cover; the "
                "longer projects continue past their first page."
            )
    summary = f"Wrote {output_path} ({len(projects)} project(s)"
    if pages:
        summary += f", {pages} page(s)"
    summary += "."
    if cover["model_tag"]:
        summary += f" Grader model {cover['model_tag']}."
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
