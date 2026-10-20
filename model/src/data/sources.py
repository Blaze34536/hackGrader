"""Load cached real-repository evidence for candidate dataset generation.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import os
import re
import time
import zlib
from collections.abc import Callable
from typing import Any

import detection as detector

from .config import REPO_CACHE_DIR, REPO_CACHE_HOURS
from .labels import LabelManifest
from .rendering import core
from .validation import slug_exposes_label


def cached_repository(url: str, fetcher: Callable[[str], str]) -> str:
    """Use a fresh repository dump cache or fall back to its last good copy."""
    path = os.path.join(REPO_CACHE_DIR, re.sub(r"[^A-Za-z0-9._-]+", "-", url) + ".txt")
    readable = os.path.exists(path)
    fresh = readable and time.time() - os.path.getmtime(path) < REPO_CACHE_HOURS * 3600
    if fresh:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    fetched = fetcher(url)
    if fetched and not str(fetched).startswith(("Error", "Repository ")):
        os.makedirs(REPO_CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(fetched)
        return fetched
    if readable:
        print(f"[i] {url}: live read failed, using the cached dump")
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    return fetched


def repos_reachable() -> bool:
    """Check GitHub reachability once before attempting every repository URL."""
    try:
        from repositories.reader import github_reachable
    except (ImportError, SyntaxError):
        return True
    return github_reachable()


def repo_fetcher():
    """Load the shared GitHub reader or report that it is unavailable."""
    try:
        from repositories.reader import read_repository
    except (ImportError, SyntaxError) as exc:
        print(f"[!] Live repository reading unavailable ({type(exc).__name__}: {exc}).")
        return None
    return read_repository


def repo_rows(
    path: str, fetcher: Callable[[str], str], labels: LabelManifest | None = None
) -> list[dict[str, Any]]:
    """Fetch listed repositories and retain their URL in every evidence row.

    When a verified-label manifest is supplied, a repository without an accepted label
    is excluded instead of being given a style-derived target, and the accepted entry
    travels with the row so the target can be the percentage that entry measured. Verified
    labels are the only repository supervision the fail-closed standard allows.
    """
    if not os.path.exists(path):
        print(f"[!] Repo list {path} not found.")
        return []
    with open(path, encoding="utf-8") as handle:
        urls = [url for url in (line.split("#", 1)[0].strip() for line in handle) if url]
    rows = []
    excluded = 0
    revealing = 0
    for position, url in enumerate(urls, start=1):
        if slug_exposes_label(url):
            # A slug such as ``project-AI`` hands the model its label in the prompt, and the row
            # contract rejects that row. Dropping it here keeps one such name from failing the build.
            revealing += 1
            continue
        verified_entry = labels.entry_for(url) if labels is not None else None
        verified = verified_entry["label"] if verified_entry else None
        if labels is not None and verified is None:
            excluded += 1
            continue
        print(f"[i] reading {url} ({position}/{len(urls)})", flush=True)
        fetched = cached_repository(url, fetcher)
        if not fetched or fetched.startswith(("Error", "Repository ")):
            print(f"[!] Skipping {url}: {str(fetched)[:90]}")
            continue
        analysis = detector.analyze(fetched)
        print(
            f"[i] {url}: detector AI {analysis['ai_score']}% / quality "
            f"{analysis['quality_score']}% over {analysis['coverage']['files_shown']} "
            f"file(s), {analysis['metrics']['lines']} line(s)"
        )
        rows.append(core(
            fetched, url, seed=zlib.crc32(url.encode("utf-8")), analysis=analysis,
            verified_label=verified, verified_entry=verified_entry, kind="repository",
            source_type="real_repository", source_groups=[url],
        ))
    if labels is not None:
        print(
            f"[i] repositories: {len(rows)} of {len(urls)} row(s) carry a verified "
            f"label; {excluded} without an accepted label were excluded (fail closed); "
            f"{revealing} whose slug names the class were excluded."
        )
        return rows
    print(f"[i] repositories: {len(rows)} of {len(urls)} row(s)")
    if len(rows) < len(urls):
        print(f"[!] {len(urls) - len(rows)} repository(ies) in {path} could not be read.")
    return rows
