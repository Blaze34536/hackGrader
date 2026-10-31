"""Render scores.json into one printable HackGrader PDF per team.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from grading.reporting import Reporting
from repositories.urls import repo_slug

from .context import build_context, cover_context
from .renderer import overlong_lines, write_pdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "scores.json"
REPORTS_DIR = PROJECT_ROOT / "reports"


def team_filename(name: str) -> str:
    """Name a team's PDF after its GitHub owner/repo, with the slash made filename-safe."""
    return re.sub(r"[^\w.-]+", "-", repo_slug(str(name))) + ".pdf"


def write_team_report(
    name: str, details: Any, reports_dir: Path, source: Path | str | None = None
) -> Path | None:
    """Write one team's PDF into ``reports_dir`` and return its path, or ``None`` on failure."""
    # Annotation rewrites the record in place; the caller's copy is still the raw result.
    projects = build_context(Reporting().annotate_all({name: copy.deepcopy(details)}))
    path = Path(reports_dir) / team_filename(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_pdf(path, projects, cover_context(source, projects)):
        return None
    overlong = overlong_lines(projects)
    if overlong:
        print(f"[!] {path.name}: {overlong} line(s) still cross the margin; check the wrap constants.")
    return path


def main(argv: list[str] | None = None) -> int:
    """Render a scores JSON file into one PDF per team inside a reports folder."""
    parser = argparse.ArgumentParser(description="Render annotated scores as per-team PDF reports.")
    parser.add_argument(
        "input", nargs="?", type=Path, default=DEFAULT_INPUT, help="scores JSON file"
    )
    parser.add_argument(
        "output", nargs="?", type=Path, default=REPORTS_DIR, help="folder for the team PDFs"
    )
    args = parser.parse_args(argv)
    if not args.input.exists():
        print(f"Error: {args.input} was not found. Run the grading pipeline first.")
        return 1
    with open(args.input, encoding="utf-8") as file:
        scores = json.load(file)
    failed = 0
    for name, details in scores.items():
        path = write_team_report(name, details, args.output, args.input)
        if path is None:
            failed += 1
        else:
            print(f"Wrote {path}")
    if failed:
        print(f"Error: PDF generation failed for {failed} team(s); the renderer reported the problem above.")
        return 1
    print(f"Wrote {len(scores)} team report(s) to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
