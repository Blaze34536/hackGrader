"""Render repository path counts as concise human-readable coverage summaries.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from typing import Any


def describe_coverage(coverage: dict[str, Any]) -> str:
    """Summarize discovered, fetched, and reproduced file counts when available."""
    if not coverage:
        return ""
    shown = coverage.get("files_shown")
    found = coverage.get("files_found", shown)
    fetched = coverage.get("files_fetched", shown)
    if shown is None:
        return ""
    parts = [f"{shown} file(s) shown"]
    if fetched is not None and fetched != shown:
        parts.append(f"{fetched} fetched")
    if found is not None and found != fetched:
        parts.append(f"{found} found")
    if coverage.get("truncated"):
        parts.append("some evidence omitted")
    return "; ".join(parts)
