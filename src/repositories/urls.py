"""Parse GitHub repository URLs and load local environment settings.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

GITHUB_HOSTS = {"github.com", "www.github.com"}


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE settings without overriding the process environment."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if value[:1] in {"'", '"'}:
            quote = value[0]
            value = value[1:].split(quote, 1)[0]
        else:
            value = value.split(" #", 1)[0].rstrip()
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(Path(__file__).resolve().parents[2] / ".env")


def is_github_url(url: str) -> bool:
    """Return whether the URL uses a supported GitHub hostname."""
    if not url:
        return False
    candidate = url.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    return urlparse(candidate).netloc.lower() in GITHUB_HOSTS


def parse_github_url(url: str) -> tuple[str, str]:
    """Return the owner and repository path components from a GitHub URL."""
    candidate = url.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = urlparse(candidate)
    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) >= 2:
        return parts[0], parts[1].removesuffix(".git").removesuffix("#")
    return "", ""


def repo_slug(url: str) -> str:
    """Return the canonical owner/repository identifier for a GitHub URL."""
    owner, repo = parse_github_url(url)
    if owner and repo:
        return f"{owner}/{repo}"
    return url.strip().rstrip("/") or "unknown-repository"
