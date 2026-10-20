"""Provide synchronous access to bounded GitHub repository evidence dumps.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import urllib.error
import urllib.request
from collections.abc import Awaitable

import aiohttp

from .collection import collect_repository
from .settings import READ_TIMEOUT_SECONDS
from .urls import is_github_url, parse_github_url


def _run(coro: Awaitable[str]) -> str:
    """Run an asynchronous read from synchronous callers, including active event loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def read_repository(url: str) -> str:
    """Read prioritized public GitHub files into a labelled repository dump."""
    if not is_github_url(url):
        return f"Error: Invalid GitHub URL '{url}'. Only github.com repositories can be read."
    owner, repo = parse_github_url(url)
    if not owner or not repo:
        return f"Error: Invalid GitHub URL '{url}'. Expected https://github.com/<owner>/<repo>."
    try:
        return _run(asyncio.wait_for(_async_read_repository(url), timeout=READ_TIMEOUT_SECONDS))
    except TimeoutError:
        return (
            f"Error: Timed out after {READ_TIMEOUT_SECONDS}s while reading {owner}/{repo}. "
            "GitHub may be rate limiting this machine."
        )
    except (aiohttp.ClientError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return f"Error fetching repository {owner}/{repo}: {type(exc).__name__}: {exc}"


def github_reachable(timeout: float = 5.0) -> bool:
    """Return whether this process can reach GitHub's API within a short timeout."""
    request = urllib.request.Request(
        "https://api.github.com/rate_limit", headers={"User-Agent": "HackGrader/0.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


async def _async_read_repository(url: str) -> str:
    """Fetch repository metadata and prioritized source files within one session."""
    import os

    headers = {"User-Agent": "HackGrader/0.1"}
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=20)
    semaphore = asyncio.Semaphore(8)
    failures: dict[str, str] = {}
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        return await collect_repository(url, session, semaphore, failures)


__all__ = ["github_reachable", "read_repository"]
