"""Recover a repository's own files when GitHub truncates its recursive tree.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .selection import is_candidate
from .settings import MAX_FILES, MAX_TREE_DEPTH, MAX_TREE_REQUESTS

Fetcher = Callable[..., Awaitable[Any]]


def _is_directory_candidate(path: str) -> bool:
    """Apply the file rules to a directory by checking it as a path prefix."""
    return is_candidate(f"{path}/")


async def walk_directories(
    fetch: Fetcher,
    owner: str,
    repo: str,
    *,
    tree_paths: list[str] = (),
    max_requests: int = MAX_TREE_REQUESTS,
    max_depth: int = MAX_TREE_DEPTH,
    max_files: int = MAX_FILES,
) -> tuple[list[str], bool]:
    """List directories shallowest-first until the project's own files are found.

    GitHub truncates ``git/trees?recursive=1`` for very large repositories, and where the
    walk stops can leave nothing but a stray root file: one graded repository kept
    ``.gitignore`` while 37k paths under an accidentally committed Cloud SDK were rejected
    by the ignored-directory rule, and the judge then scored 85% AI involvement from eight
    lines of ``.gitignore``.

    The walk keeps the paths the tree did return and adds what the directory listings
    show, breadth-first, so the project's own shallow files are found before a deep
    dependency tree is explored. It stops at the request budget, at the depth limit, and
    once enough candidate files are known. Returns the candidate paths and whether the root
    listing answered at all, because a repository the tree cannot describe and the contents
    API cannot list has not been read.
    """
    found = [path for path in tree_paths if is_candidate(path)]
    known = set(found)
    queue: list[tuple[str, int]] = [("", 0)]
    requests = 0
    listed_root = False
    while queue and requests < max_requests and len(found) < max_files:
        path, depth = queue.pop(0)
        if depth > max_depth:
            continue
        requests += 1
        listing = await fetch(
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}", is_json=True
        )
        if not isinstance(listing, list):
            continue
        if not path:
            listed_root = True
        for entry in listing:
            entry_path = entry.get("path") or ""
            if not entry_path or entry_path in known:
                continue
            if entry.get("type") == "dir":
                if _is_directory_candidate(entry_path):
                    known.add(entry_path)
                    queue.append((entry_path, depth + 1))
            elif is_candidate(entry_path):
                known.add(entry_path)
                found.append(entry_path)
    return found, listed_root
