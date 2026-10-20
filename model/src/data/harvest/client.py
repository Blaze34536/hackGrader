"""Talk to the GitHub REST API through a counted, injectable, dependency-free client.

Every harvest call goes through one method so the request budget cannot be bypassed, and the
transport is a plain function returning ``(status, headers, payload)``. That shape is what
lets the whole harvester be tested offline against a fixed set of recorded responses instead
of the network, and it keeps the harvest runnable in the model environment, which does not
install the application's ``aiohttp``-based reader.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from .budget import RequestBudget

API_ROOT = "https://api.github.com"
RAW_ROOT = "https://raw.githubusercontent.com"
USER_AGENT = "HackGrader/0.1"
MAX_ATTEMPTS = 3

# (status, headers, payload). The payload is parsed JSON when the response was JSON.
FetchResult = tuple[int, dict[str, str], Any]
Fetcher = Callable[[str], FetchResult]


class GithubError(RuntimeError):
    """Raised when GitHub rejects a request for a reason retrying cannot fix."""


def github_token() -> str | None:
    """Return a GitHub token from the environment, loading the repository ``.env`` first.

    The application reads its token from ``GITHUB_TOKEN`` and its URL module loads a
    repository-root ``.env`` on import, so importing it here gives a harvest run the same
    credentials the grader uses. A missing token is not an error: an anonymous harvest is
    slower and smaller, and the rate limit the API reports is recorded either way.
    """
    if not os.environ.get("GITHUB_TOKEN"):
        # Importing the URL module loads the repository-root .env; it is absent when the
        # model workspace is used without the application tree, which is not an error here.
        with contextlib.suppress(ImportError):
            import repositories.urls  # noqa: F401
    return os.environ.get("GITHUB_TOKEN") or None


def urllib_fetch(url: str, *, token: str | None = None, timeout: float = 25.0) -> FetchResult:
    """Fetch one URL with the standard library and return status, headers, and payload."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
            received = {key: value for key, value in response.headers.items()}
    except urllib.error.HTTPError as error:
        body = error.read()
        status = error.code
        received = {key: value for key, value in error.headers.items()}
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, {}, None
    text = body.decode("utf-8", errors="ignore")
    try:
        return status, received, json.loads(text)
    except json.JSONDecodeError:
        return status, received, text


def _wait_seconds(headers: dict[str, str], attempt: int) -> float:
    """Return how long to wait before retrying a throttled or failing request."""
    lowered = {str(key).casefold(): value for key, value in headers.items()}
    retry_after = lowered.get("retry-after")
    if retry_after and str(retry_after).isdigit():
        return min(float(retry_after), 60.0)
    reset = lowered.get("x-ratelimit-reset")
    if lowered.get("x-ratelimit-remaining") == "0" and reset and str(reset).isdigit():
        wait = int(reset) - int(time.time())
        if wait > 0:
            return min(float(wait), 60.0)
    return min(2.0 * (attempt + 1), 30.0)


class GithubClient:
    """A counted GitHub client: metered API calls, unmetered content reads, both recorded."""

    def __init__(
        self,
        budget: RequestBudget,
        *,
        fetch: Fetcher | None = None,
        token: str | None = None,
        timeout: float = 25.0,
        pause: float = 0.0,
        attempts: int = MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Wrap a budget with a transport, a token, and the retry policy."""
        self.budget = budget
        self.timeout = timeout
        self.pause = pause
        self.attempts = max(attempts, 1)
        self._sleep = sleep
        self._token = token
        self._fetch = fetch
        self.misses: list[str] = []
        self.errors: list[str] = []

    def _call(self, url: str) -> FetchResult:
        """Perform one request through the injected or standard-library transport."""
        self.budget.spend()
        if self.pause > 0:
            self._sleep(self.pause)
        if self._fetch is not None:
            return self._fetch(url)
        if self._token is None:
            self._token = github_token()
        return urllib_fetch(url, token=self._token, timeout=self.timeout)

    def get(self, url: str) -> Any | None:
        """Return a JSON payload, or ``None`` when the resource is missing or unavailable.

        A 422 is raised instead of swallowed: it is how GitHub reports a query it cannot run,
        such as the unknown app slug that would otherwise look like an agent with no merged
        pull requests and quietly halve a cohort.
        """
        for attempt in range(self.attempts):
            status, headers, payload = self._call(url)
            self.budget.observe(headers)
            if status == 200:
                return payload
            if status == 422:
                message = payload.get("message") if isinstance(payload, dict) else payload
                raise GithubError(f"GitHub rejected {url}: {message}")
            if status in (403, 429) or status >= 500 or status == 0:
                detail = f"HTTP {status}" if status else "no response"
                self.errors.append(f"{url}: {detail}")
                if attempt + 1 < self.attempts:
                    self._sleep(_wait_seconds(headers, attempt))
                continue
            self.misses.append(f"{url}: HTTP {status}")
            return None
        return None

    def raw(self, url: str) -> str | None:
        """Return raw file text from the content CDN, which is outside the API quota."""
        if self._fetch is not None:
            status, _, payload = self._fetch(url)
            self.budget.note_raw()
            return payload if status == 200 and isinstance(payload, str) else None
        self.budget.note_raw()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8", errors="ignore")
        except (urllib.error.URLError, TimeoutError, OSError):
            return None

    def line_count(self, owner: str, repo: str, branch: str, path: str) -> int:
        """Return the current line count of one file, or zero when it cannot be read.

        The bound in the last-touch rule needs the file's length now, not the length the
        attributed commit saw, so the content is read from the CDN rather than from an
        endpoint that would spend the API budget on it.
        """
        quoted = urllib.parse.quote(path)
        text = self.raw(f"{RAW_ROOT}/{owner}/{repo}/{branch}/{quoted}")
        return len(text.splitlines()) if text else 0


__all__ = [
    "API_ROOT",
    "RAW_ROOT",
    "FetchResult",
    "Fetcher",
    "GithubClient",
    "GithubError",
    "github_token",
    "urllib_fetch",
]
