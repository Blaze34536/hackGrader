"""Extract file locations and structural metrics from repository evidence.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import ast
import re
import warnings
from itertools import pairwise

from .base import BOILERPLATE_FILES, DOCUMENTATION_EXTS, SOURCE_EXTS
from .signals import (
    BARE_SITE,
    COMMENT_LINE,
    COUNTED_PATTERNS,
    DEF_PATTERN,
    DOCSTRING_PATTERN,
    FILE_BLOCK,
    FILE_HEADER,
    JSDOC_PATTERN,
    README_BLOCK,
    README_CASUAL,
    README_FIRST_PERSON,
    README_TYPOS,
    SPACED_SITE,
    STOPWORDS,
    TYPED_DEF_PATTERN,
)


def files_in(code: str) -> list[str]:
    """Return file paths explicitly labeled in a repository dump."""
    return [match.group(1).strip() for match in FILE_HEADER.finditer(code)]


def _file_at(code: str, offset: int) -> str:
    """Find the file label active at a character offset in a dump."""
    name = "the repository"
    for match in FILE_HEADER.finditer(code):
        if match.start() > offset:
            break
        name = match.group(1).strip()
    return name


def strip_boilerplate(code: str) -> str:
    """Blank the bodies of tool-generated configuration files such as .gitignore.

    Their headers stay so offsets and file attribution still resolve, but their
    template comments and banners are not read as authorship evidence.
    """

    def blank(match: re.Match) -> str:
        name = match.group(1).strip().lower()
        if name.rsplit("/", 1)[-1] in BOILERPLATE_FILES:
            return match.group(0)[: match.start(2) - match.start(0)]
        return match.group(0)

    return FILE_BLOCK.sub(blank, code)


def _in_documentation(code: str, offset: int) -> bool:
    """Return whether an offset falls inside a documentation file block."""
    return _file_at(code, offset).lower().endswith(DOCUMENTATION_EXTS)


def identifiers(code: str) -> list[str]:
    """Return unique declaration names found in supported source text."""
    names = re.findall(
        r"^\s*(?:def|class|function|func|public|private|protected|static|const|export)\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)", code, re.MULTILINE,
    )
    return [name for name in dict.fromkeys(names) if len(name) > 2]


def _count(pattern: re.Pattern, code: str) -> int:
    """Count regex matches in source text."""
    return len(pattern.findall(code))


def _source_blocks(code: str) -> list[tuple[str, str]]:
    """Return source-file blocks, or the unlabelled text as one source block."""
    blocks = FILE_BLOCK.findall(code)
    if not blocks:
        return [("", code)]
    return [(name, text) for name, text in blocks if name.strip().lower().endswith(SOURCE_EXTS)]


def _shape_metrics(code: str) -> dict[str, int]:
    """Measure source lines, formatting, and Python syntax across source blocks."""
    lines: list[str] = []
    code_lines: list[str] = []
    python_files = python_errors = 0
    blocks = _source_blocks(code)
    for name, text in blocks:
        for line in text.split("\n"):
            if line.strip():
                lines.append(line)
                if not COMMENT_LINE.match(line):
                    code_lines.append(line)
        if name.strip().lower().endswith(".py"):
            python_files += 1
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ast.parse(text)
            except SyntaxError:
                python_errors += 1
    body = "\n".join(code_lines)
    slips = len(BARE_SITE.findall(body))
    return {
        "source_files": len(blocks), "source_lines": len(lines),
        "code_lines": len(code_lines), "comment_lines": len(lines) - len(code_lines),
        "spacing_slips": slips, "spacing_sites": slips + len(SPACED_SITE.findall(body)),
        "tab_lines": sum(line.startswith("\t") for line in lines),
        "trailing_lines": sum(line != line.rstrip() for line in lines),
        "python_files": python_files, "python_syntax_errors": python_errors,
    }


def _restated_docstrings(code: str) -> tuple[int, int]:
    """Count Python docstrings whose content repeats the function name."""
    total = restated = 0
    for match in DOCSTRING_PATTERN.finditer(code):
        total += 1
        name_words = {word.lower() for word in re.split(r"_+", match.group(1)) if word}
        content = {
            word.lower() for word in re.findall(r"[A-Za-z]+", match.group(2))
            if word.lower() not in STOPWORDS
        }
        if content and (content <= name_words or len(content & name_words) / len(content) >= 0.5):
            restated += 1
    return restated, total


def _longest_function(code: str) -> tuple[int, str]:
    """Estimate the longest function by counting until the next declaration."""
    lines = code.split("\n")
    longest, longest_name, start, name = 0, "", None, ""
    for index, line in enumerate(lines):
        if re.match(r"^\s*(?:def|function|func|public|private|protected)\s+\w+", line):
            if start is not None:
                longest, longest_name = max((longest, longest_name), (index - start, name))
            start, name = index, line.strip()[:60]
    if start is not None:
        longest, longest_name = max((longest, longest_name), (len(lines) - start, name))
    return longest, longest_name


def _repeated_block(code: str) -> int:
    """Return the size of the longest repeated adjacent line pair."""
    lines = [line.strip() for line in code.split("\n") if line.strip()]
    counts: dict[tuple[str, str], int] = {}
    for first, second in pairwise(lines):
        if len(first) >= 12:
            counts[(first, second)] = counts.get((first, second), 0) + 1
    return max((len(key) for key, count in counts.items() if count >= 4), default=0)


def _readme_metrics(code: str) -> dict[str, int]:
    """Measure README length, structure, first-person phrasing, and informal markers."""
    match = README_BLOCK.search(code)
    if not match:
        return dict.fromkeys((
            "readme_chars", "readme_headings", "readme_bullets",
            "readme_first_person", "readme_casual", "readme_typos",
        ), 0)
    text = match.group(1)
    return {
        "readme_chars": len(text),
        "readme_headings": len(re.findall(r"^#{1,6} ", text, re.MULTILINE)),
        "readme_bullets": len(re.findall(r"^[ \t]*[-*+] ", text, re.MULTILINE)),
        "readme_first_person": len(README_FIRST_PERSON.findall(text)),
        "readme_casual": len(README_CASUAL.findall(text)),
        "readme_typos": len(README_TYPOS.findall(text)),
    }


def metrics(code: str) -> dict[str, int | str]:
    """Collect the full feature set used by detector scoring and prompt measurements."""
    text = [line for line in code.split("\n") if line.strip() and not line.startswith("--- FILE:")]
    longest, longest_name = _longest_function(code)
    restated, documented = _restated_docstrings(code)
    return {
        "files": len(files_in(code)), "lines": len(text), **_shape_metrics(code),
        **_readme_metrics(code), "chars": len(code), "functions": _count(DEF_PATTERN, code),
        "docstrings": documented, "restated_docstrings": restated,
        "typed_functions": _count(TYPED_DEF_PATTERN, code),
        "jsdoc_blocks": _count(JSDOC_PATTERN, code),
        "longest_function": longest, "longest_function_name": longest_name,
        "repeated_block": _repeated_block(code),
        **{name: _count(pattern, code) for name, pattern in COUNTED_PATTERNS.items() if name != "docs_comments"},
    }
