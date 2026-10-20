"""Render detector measurements and repository evidence into judge prompts.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from typing import Any

from .prompts import (
    FILE_NAME_HEADER,
    FILE_TRUNCATED_HEADER,
    FILE_TRUNCATION_MARKER,
    FILES_NOT_FETCHED_HEADER,
    FILES_NOT_INCLUDED_HEADER,
    REPO_CONTENT_BEGIN,
    REPO_CONTENT_END,
    REPORT_INSTRUCTION,
)


def measurement_block(analysis: dict[str, Any], repo_block: str | None = None) -> str:
    """Format detector measurements and distinguish repository/view coverage."""
    metrics = analysis["metrics"]
    coverage = analysis["coverage"]
    found = coverage.get("files_found", coverage["files_shown"])
    fetched = coverage.get("files_fetched", coverage["files_shown"])
    not_fetched_match = FILES_NOT_FETCHED_HEADER.search(repo_block or "")
    not_fetched = not_fetched_match.group(1).strip() if not_fetched_match else ""
    not_included_match = FILES_NOT_INCLUDED_HEADER.search(repo_block or "")
    not_included = not_included_match.group(1).strip() if not_included_match else ""
    reproduced = (
        len(set(FILE_NAME_HEADER.findall(repo_block)))
        if repo_block is not None
        else coverage["files_shown"]
    )
    partial = (
        coverage.get("truncated", False)
        or bool(FILE_TRUNCATED_HEADER.search(repo_block or ""))
        or reproduced < fetched
        or (repo_block is not None and FILE_TRUNCATION_MARKER in repo_block)
    )
    coverage_text = (
        f"{found} relevant path(s) found; {fetched} fetched; "
        f"{reproduced} reproduced in this batch"
    )
    if not_included:
        coverage_text += f"; fetched but not in this batch: {not_included}"
    if not_fetched:
        coverage_text += f"; not fetched: {not_fetched}"
    if partial:
        coverage_text += "; some files or file contents are omitted"

    lines = [
        (
            "FACTUAL PROMPT CONTEXT FROM THE FILES IN THIS EVIDENCE BATCH "
            "(measurements only, not AI-authorship or quality scores; repository-wide counts "
            "are reported separately):"
        ),
        f"- Repository coverage: {coverage_text}",
        (
            f"- Source files: {metrics['source_files']}"
            f" | source lines: {metrics['source_lines']}"
            f" | non-blank lines in total: {metrics['lines']}"
            f" | characters: {metrics['chars']}"
        ),
        (
            f"- Definitions: {metrics['functions']}"
            f" | carrying a docstring: {metrics['docstrings']}"
            f" | docstrings that only restate the name: {metrics['restated_docstrings']}"
        ),
        f"- Type annotations: {metrics['typed_functions']} | JSDoc blocks: {metrics['jsdoc_blocks']}",
        (
            f"- TODO/FIXME notes: {metrics['todos']}"
            f" | debug prints: {metrics['debug_prints']}"
            f" | commented-out code lines: {metrics['commented_out']}"
        ),
        (
            f"- Bare except blocks: {metrics['bare_excepts']}"
            f" | hard-coded secrets: {metrics['credentials']}"
            f" | localhost endpoints: {metrics['localhost']}"
            f" | interpolated SQL: {metrics['sql_interpolation']}"
        ),
        f"- Hard-coded local paths: {metrics['local_paths']} | emoji in code or comments: {metrics['emoji']}",
        f"- Source code lines: {metrics['code_lines']} | source lines carrying a comment: {metrics['comment_lines']}",
        # Formatting counts are deliberately absent. The prompt states that indentation and
        # whitespace are not authorship evidence, so publishing "Tab-indented lines: 0" only
        # offered the judge a number to read as machine-like discipline: a project indented
        # with spaces scored like a generated one, and the count appeared in reports.
        f"- Python files that do not parse: {metrics['python_syntax_errors']} of {metrics['python_files']}",
    ]
    if metrics["longest_function"]:
        longest = f"- Longest single function: {metrics['longest_function']} lines"
        if metrics["longest_function_name"]:
            longest += f" (`{metrics['longest_function_name']}`)"
        lines.append(longest)
    # The README is measured last and named as the project's own description: it is context
    # for reading the code above, not a second source of evidence about who wrote it.
    lines.append(
        f"- README (the project's own description of itself, context only): "
        f"{metrics['readme_chars']} characters"
        f" | headings: {metrics['readme_headings']}"
        f" | bullet lines: {metrics['readme_bullets']}"
        f" | first-person references: {metrics['readme_first_person']}"
        f" | casual markers or typos: {metrics['readme_casual'] + metrics['readme_typos']}"
    )
    return "\n".join(lines)


def build_user_message(
    question: str,
    repo_block: str,
    *,
    url: str,
    analysis: dict[str, Any] | None = None,
) -> str:
    """Place one repository view inside clear delimiters and append measurements."""
    measurements = f"\n\n{measurement_block(analysis, repo_block)}" if analysis else ""
    return (
        f"{question.strip()}\n\n"
        f"Repository content for {url} is reproduced between the markers below.\n\n"
        f"{REPO_CONTENT_BEGIN}\n{repo_block}\n"
        f"{REPO_CONTENT_END}\n"
        f"{measurements}\n\n{REPORT_INSTRUCTION}"
    )
