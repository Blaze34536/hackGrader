"""Fetch useful repository files and serialize truthful coverage metadata.

A recursive tree GitHub truncated is replaced by a bounded directory walk, so a repository
is never scored from a stray root file the tree happened to list first.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import asyncio

import aiohttp

from contracts.prompts import FILE_TRUNCATION_MARKER
from contracts.repository_dump import OMITTED_FILES_MARKER

from .discovery import walk_directories
from .selection import (
    is_candidate,
    is_documentation,
    looks_binary,
    path_listing,
    redact_payloads,
    selection_priority,
)
from .settings import MAX_DOC_CHARS, MAX_FILES, MAX_PER_FILE_CHARS, MAX_TOTAL_CHARS
from .transport import make_fetcher
from .urls import parse_github_url


def _failure(url: str, failures: dict[str, str]) -> str:
    """Format the last response failure for a repository endpoint."""
    detail = failures.get(url)
    suffix = f" ({detail})" if detail else ""
    return suffix


def _render_dump(
    owner: str,
    repo: str,
    branch: str,
    all_candidates: list[str],
    fetched: list[str],
    included: list[str],
    not_included: list[str],
    not_fetched: list[str],
    blocks: list[str],
    truncated: bool,
) -> str:
    """Build the shared dump header with complete discovery and fetch counts."""
    header = (
        f"REPOSITORY: {owner}/{repo}\n"
        f"DEFAULT BRANCH: {branch}\n"
        f"FILES FOUND: {len(all_candidates)}\n"
        f"FILES FETCHED: {len(fetched)}\n"
        f"FILES INCLUDED ({len(included)}): {path_listing(included)}\n"
        f"FILES TRUNCATED: {'yes' if truncated else 'no'}"
    )
    if not_included:
        header += f"\nFILES NOT INCLUDED ({len(not_included)}): {path_listing(not_included)}"
    if not_fetched:
        header += f"\nFILES NOT FETCHED ({len(not_fetched)}): {path_listing(not_fetched)}"
    footer = f"\n\n{OMITTED_FILES_MARKER}" if not_included or not_fetched else ""
    return header + "".join(blocks) + footer


async def collect_repository(
    url: str,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    failures: dict[str, str],
) -> str:
    """Read repository metadata and a prioritized, bounded set of file contents."""
    owner, repo = parse_github_url(url)
    fetch = make_fetcher(session, semaphore, failures)
    repo_api = f"https://api.github.com/repos/{owner}/{repo}"
    repo_data = await fetch(repo_api, is_json=True)
    if not repo_data:
        detail = _failure(repo_api, failures)
        return (
            f"Error: Could not fetch repository data for {owner}/{repo}{detail}. "
            "The repository may be private, renamed, or GitHub is rate limiting this machine."
        )

    branch = repo_data.get("default_branch", "main")
    tree_api = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    tree_data = await fetch(tree_api, is_json=True)
    if not tree_data:
        detail = _failure(tree_api, failures)
        return f"Error: Could not fetch repository tree for branch '{branch}'{detail}."

    tree_paths = [
        item.get("path", "")
        for item in tree_data.get("tree", [])
        if item.get("type") == "blob"
    ]
    if tree_data.get("truncated"):
        # A truncated tree is a partial listing, and where GitHub stops can hide everything
        # the project wrote: one graded repository kept only `.gitignore`, because every
        # other listed path sat under a vendored Cloud SDK the selector ignores. List
        # directories rather than score a repository from a stray root file.
        contents_api = f"https://api.github.com/repos/{owner}/{repo}/contents/"
        discovered, listed_root = await walk_directories(
            fetch, owner, repo, tree_paths=tree_paths
        )
        if not listed_root:
            detail = _failure(contents_api, failures)
            return (
                f"Error: GitHub truncated the file tree for {owner}/{repo} and its "
                f"directory listing could not be read{detail}. A partial tree would score "
                "the repository from a handful of stray files, so no report was produced."
            )
        all_candidates = sorted(dict.fromkeys(discovered), key=selection_priority)
    else:
        all_candidates = sorted(
            (path for path in tree_paths if is_candidate(path)), key=selection_priority
        )
    candidates = all_candidates[:MAX_FILES]
    if not candidates:
        return f"Repository {owner}/{repo} contains no readable code files."

    included: list[str] = []
    fetched: list[str] = []
    not_included: list[str] = []
    not_fetched: list[str] = list(all_candidates[MAX_FILES:])
    blocks: list[str] = []
    current_chars = 0
    doc_chars = 0
    truncated = bool(tree_data.get("truncated")) or len(all_candidates) > len(candidates)
    stop_fetching = False

    for start in range(0, len(candidates), 6):
        batch = candidates[start:start + 6]
        texts = await asyncio.gather(*(
            fetch(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}")
            for path in batch
        ))
        for offset, (path, text) in enumerate(zip(batch, texts, strict=True)):
            prefix = f"\n\n--- FILE: {path} ---\n"
            if text is None:
                not_fetched.append(path)
                truncated = True
                continue
            fetched.append(path)
            if looks_binary(text):
                # Record the omission instead of reproducing the bytes: the grader both
                # wastes its answer on them and can cite them as authorship evidence.
                not_included.append(path)
                truncated = True
                continue
            # An encoded payload inside a readable file is the same hazard in text form: an
            # inline data URI or a base64 icon is quoted back as if it were code. Redacting
            # it before the per-file budget also spends that budget on source instead.
            text = redact_payloads(text)
            documentation = is_documentation(path)
            if (
                documentation
                and doc_chars + len(prefix) + min(len(text), MAX_PER_FILE_CHARS)
                > MAX_DOC_CHARS
            ):
                not_included.append(path)
                truncated = True
                continue
            if not text.strip():
                if current_chars + len(prefix) <= MAX_TOTAL_CHARS:
                    blocks.append(prefix)
                    included.append(path)
                    current_chars += len(prefix)
                    doc_chars += len(prefix) if documentation else 0
                else:
                    not_included.append(path)
                    truncated = True
                continue

            content = text[:MAX_PER_FILE_CHARS]
            if len(text) > MAX_PER_FILE_CHARS:
                content += FILE_TRUNCATION_MARKER
                truncated = True
            available = MAX_TOTAL_CHARS - current_chars
            if len(prefix) + len(content) > available:
                body_budget = available - len(prefix)
                marker_size = len(FILE_TRUNCATION_MARKER)
                if body_budget > marker_size:
                    content = text[:body_budget - marker_size] + FILE_TRUNCATION_MARKER
                    blocks.append(prefix + content)
                    included.append(path)
                    current_chars += len(prefix) + len(content)
                else:
                    not_included.append(path)
                truncated = True
                for pending, pending_text in zip(batch[offset + 1:], texts[offset + 1:], strict=True):
                    if pending_text is None:
                        not_fetched.append(pending)
                    else:
                        fetched.append(pending)
                        not_included.append(pending)
                not_fetched.extend(candidates[start + len(batch):])
                stop_fetching = True
                break
            blocks.append(prefix + content)
            included.append(path)
            current_chars += len(prefix) + len(content)
            doc_chars += len(prefix) + len(content) if documentation else 0
        if stop_fetching:
            break

    included_set = set(included)
    fetched = list(dict.fromkeys(fetched))
    not_included = list(dict.fromkeys(path for path in not_included if path not in included_set))
    not_fetched = list(dict.fromkeys(path for path in not_fetched if path not in fetched))
    truncated = truncated or bool(not_included) or bool(not_fetched)
    if not blocks:
        detail = next(iter(failures.values()), "")
        suffix = f" ({detail})" if detail else ""
        return f"Repository {owner}/{repo} returned no readable file contents{suffix}."
    return _render_dump(
        owner, repo, branch, all_candidates, fetched, included,
        not_included, not_fetched, blocks, truncated,
    )
