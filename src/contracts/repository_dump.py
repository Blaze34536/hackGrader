"""Parse and represent metadata and file blocks from repository dumps.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .prompts import (
    FILE_NAME_HEADER,
    FILES_FETCHED_HEADER,
    FILES_FOUND_HEADER,
    FILES_INCLUDED_HEADER,
    FILES_INCLUDED_NAMES_HEADER,
    FILES_NOT_FETCHED_HEADER,
    FILES_NOT_INCLUDED_HEADER,
    FILES_TRUNCATED_HEADER,
)

SOURCE_EXTENSIONS = (
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".kts", ".go", ".rs",
    ".c", ".h", ".hpp", ".cpp", ".cs", ".rb", ".php", ".swift", ".dart", ".scala",
    ".sh", ".ps1", ".sql", ".vue", ".svelte", ".jai", ".lua", ".r", ".m", ".jl",
    ".gml", ".html", ".htm", ".css",
)
DOCUMENTATION_EXTENSIONS = (".md", ".markdown", ".txt", ".rst", ".adoc", ".mdx")
OMITTED_FILES_MARKER = "[... Remaining files omitted for context limits ...]"
UNFETCHED_CONTENT_MARKER = "[... source contents were not fetched ...]"
PATH_LIST_BUDGET = 400


@dataclass(frozen=True)
class RepositoryDump:
    """Parsed metadata and file blocks from a repository dump."""

    header: str
    files: list[tuple[str, str, str]]
    found: int
    fetched: int
    included: int
    truncated: bool
    not_fetched: list[str]
    not_included: list[str]


# Path categories, in the order a bounded view should prefer them. Source code and entry
# points come first because they are what the judge is grading. The README comes after the
# project's own tests, metadata, and configuration: it is the authors' account of their
# work, and a view that reads it before their code can score the description instead of the
# repository. Category numbers are the ranks, so both the fetch order and the excerpt order
# read from one statement of what matters.
ENTRY_POINT_CATEGORY = 0
SOURCE_CATEGORY = 1
TEST_CATEGORY = 2
METADATA_CATEGORY = 3
CONFIG_CATEGORY = 4
README_CATEGORY = 5
DOCUMENTATION_CATEGORY = 6
OTHER_CATEGORY = 7

ENTRY_POINT_NAMES = frozenset(
    {"main", "app", "index", "server", "cli", "__main__", "program", "manage"}
)
ROOT_METADATA_NAMES = frozenset(
    {"pyproject.toml", "package.json", "cargo.toml", "go.mod"}
)
CONFIG_NAMES = frozenset(
    {"requirements.txt", "dockerfile", "compose.yaml", "compose.yml", "makefile", "justfile"}
)


def view_priority(name: str) -> tuple[int, int, str]:
    """Rank paths so views spend their characters on code before prose."""
    lowered = name.lower().replace("\\", "/")
    basename = lowered.rsplit("/", 1)[-1]
    if basename.startswith("readme"):
        category = README_CATEGORY
    elif lowered.count("/") == 0 and basename in ROOT_METADATA_NAMES:
        category = METADATA_CATEGORY
    elif lowered.endswith(SOURCE_EXTENSIONS) and basename.rsplit(".", 1)[0] in ENTRY_POINT_NAMES:
        # An entry point is a code file named for its role. ``Main.storyboard`` shares the stem
        # but is interface layout, not code, so it does not take the entry-point category.
        category = ENTRY_POINT_CATEGORY
    elif (
        "/test/" in f"/{lowered}/"
        or "/tests/" in f"/{lowered}/"
        or ".test." in basename
        or ".spec." in basename
        or basename.startswith("test_")
        or basename.endswith("_test.py")
    ):
        category = TEST_CATEGORY
    elif lowered.endswith(SOURCE_EXTENSIONS):
        category = SOURCE_CATEGORY
    elif basename in CONFIG_NAMES or lowered.endswith((".json", ".yaml", ".yml", ".toml")):
        category = CONFIG_CATEGORY
    elif lowered.endswith(DOCUMENTATION_EXTENSIONS):
        category = DOCUMENTATION_CATEGORY
    else:
        category = OTHER_CATEGORY
    return category, lowered.count("/"), lowered


# What a file's content can say about who wrote it. Source code, entry points, and tests are
# what a person types. A README, a manifest or lock file, a licence, generated scaffolding,
# and build configuration are the project's own description of itself or of its tooling:
# every project of that shape carries them, whoever wrote the rest, so a sign read only from
# one of those says how the project is packaged rather than who authored it. The judge
# prompt, the citation check, and the training targets all state this rule from here.
AUTHORSHIP_CATEGORIES = frozenset({ENTRY_POINT_CATEGORY, SOURCE_CATEGORY, TEST_CATEGORY})


def carries_authorship_evidence(name: str) -> bool:
    """Return whether a path's content can support an authorship claim."""
    return view_priority(name)[0] in AUTHORSHIP_CATEGORIES


def view_limits(name: str) -> tuple[int, int]:
    """Return initial and preferred maximum excerpt sizes by file category."""
    return {
        ENTRY_POINT_CATEGORY: (1000, 2400),
        SOURCE_CATEGORY: (850, 1800),
        TEST_CATEGORY: (1000, 1800),
        METADATA_CATEGORY: (600, 1200),
        CONFIG_CATEGORY: (600, 1200),
        README_CATEGORY: (600, 1200),
        DOCUMENTATION_CATEGORY: (500, 1000),
        OTHER_CATEGORY: (500, 1000),
    }[view_priority(name)[0]]


def _parse_names(match: re.Match[str] | None) -> list[str]:
    """Parse a path-list header unless it is a compact summary marker."""
    if not match:
        return []
    value = match.group(2).strip()
    if not value or value.lower().startswith(("see ", "path list exceeds", "names unavailable")):
        return []
    return [name.strip() for name in value.split(",") if name.strip()]


def _file_blocks(dump: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Return leading metadata and parsed `(path, header, body)` records."""
    matches = list(FILE_NAME_HEADER.finditer(dump))
    if not matches:
        return dump, []
    header = dump[:matches[0].start()]
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(dump)
        block = dump[match.start():end]
        if index == len(matches) - 1:
            block = block.split(f"\n\n{OMITTED_FILES_MARKER}", 1)[0]
        line_end = block.find("\n")
        if line_end >= 0:
            blocks.append((match.group(1).strip(), block[:line_end + 1], block[line_end + 1:]))
    return header, blocks


def _parse_metadata(dump: str) -> RepositoryDump:
    """Read discovery, fetch, inclusion, and truncation counts from headers."""
    header, blocks = _file_blocks(dump)
    included_match = FILES_INCLUDED_HEADER.search(header)
    listed_names = _parse_names(FILES_INCLUDED_NAMES_HEADER.search(header))
    not_fetched = _parse_names(FILES_NOT_FETCHED_HEADER.search(header))
    not_included = _parse_names(FILES_NOT_INCLUDED_HEADER.search(header))
    by_name = {name: (file_header, body) for name, file_header, body in blocks}
    if listed_names and included_match and len(listed_names) == int(included_match.group(1)):
        blocks = [(name, *by_name[name]) for name in listed_names if name in by_name]

    included = int(included_match.group(1)) if included_match else len(blocks)
    found_match = FILES_FOUND_HEADER.search(header)
    fetched_match = FILES_FETCHED_HEADER.search(header)
    found = int(found_match.group(1)) if found_match else included
    fetched = int(fetched_match.group(1)) if fetched_match else included + len(not_included)
    if found > fetched and not not_fetched:
        not_fetched = [f"{found - fetched} additional path(s); names unavailable"]
    truncated = (
        bool(FILES_TRUNCATED_HEADER.search(header))
        or OMITTED_FILES_MARKER in dump
        or bool(not_fetched)
        or bool(not_included)
        or included < fetched
    )
    return RepositoryDump(header, blocks, found, fetched, included, truncated, not_fetched, not_included)


def _path_list(paths: list[str]) -> str:
    """Format paths under a fixed bound while retaining an explicit omission cue."""
    text = ", ".join(dict.fromkeys(paths))
    return text if len(text) <= PATH_LIST_BUDGET else "path list exceeds header budget; names unavailable"


def _base_header(meta: RepositoryDump) -> str:
    """Build repository identity and full-discovery counts for each model view."""
    lines = []
    for key in ("REPOSITORY", "DEFAULT BRANCH"):
        match = re.search(rf"(?m)^{re.escape(key)}:[ \t]*(.*)$", meta.header)
        if match:
            lines.append(f"{key}: {match.group(1).strip()}")
    lines.extend((
        f"FILES FOUND: {meta.found}", f"FILES FETCHED: {meta.fetched}",
        "FILES INCLUDED (0):", f"FILES TRUNCATED: {'yes' if meta.truncated else 'no'}",
    ))
    if meta.not_fetched:
        lines.append(f"FILES NOT FETCHED ({len(meta.not_fetched)}): {_path_list(meta.not_fetched)}")
    return "\n".join(lines) + "\n\n"


def _view_header(base: str, included: list[str], all_fetched: list[str], *, partial_file: bool = False) -> str:
    """Describe included paths, other fetched files, and any partial file excerpt."""
    included = list(dict.fromkeys(included))
    omitted = [name for name in dict.fromkeys(all_fetched) if name not in included]
    header = FILES_INCLUDED_NAMES_HEADER.sub(
        f"FILES INCLUDED ({len(included)}): {_path_list(included)}", base, count=1
    )
    if omitted:
        header = header.replace(
            "FILES TRUNCATED:",
            f"FILES NOT INCLUDED ({len(omitted)}): {_path_list(omitted)}\nFILES TRUNCATED:",
            1,
        )
    if partial_file:
        header = header.replace("FILES TRUNCATED:", "FILE EXCERPT PARTIAL: yes\nFILES TRUNCATED:", 1)
    return header
