"""Handle bounded asynchronous GitHub requests and retry diagnostics.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp


def make_fetcher(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    failures: dict[str, str],
):
    """Create an async request helper with bounded retries for transient failures."""
    async def fetch(
        fetch_url: str,
        *,
        is_json: bool = False,
        max_retries: int = 3,
    ) -> Any:
        """Fetch one URL with retries for rate limits and temporary network errors."""
        for attempt in range(max_retries):
            try:
                async with semaphore, session.get(fetch_url) as response:
                    if response.status in (403, 429):
                        failures[fetch_url] = f"HTTP {response.status}"
                        if attempt + 1 < max_retries:
                            await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    if response.status == 404:
                        failures[fetch_url] = "HTTP 404"
                        return None
                    response.raise_for_status()
                    if is_json:
                        return await response.json()
                    return await response.text(errors="ignore")
            except (TimeoutError, aiohttp.ClientError) as exc:
                failures[fetch_url] = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < max_retries:
                    await asyncio.sleep(0.75 * (attempt + 1))
        return None

    return fetch
