"""Retry transient grading failures and return setup errors promptly.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import random
import time
from typing import Any

from contracts import is_boilerplate_report, is_valid_report

from .ollama import _log


def query_with_retry(
    grader: Any,
    question: str,
    max_retries: int = 3,
    initial_delay: float = 3.0,
) -> dict[str, Any]:
    """Retry transient model errors but return configuration and input failures immediately."""
    url = grader.extract_url(question) or "the requested repository"
    if max_retries <= 0:
        return {
            "status": 500,
            "content": "Evaluation was not attempted: max_retries must be positive.",
            "analysis": None,
        }

    last_result: dict[str, Any] = {
        "status": 500,
        "content": "Evaluation did not run.",
        "analysis": None,
    }
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            delay = initial_delay * (2 ** (attempt - 2)) + random.uniform(0, 1)
            _log(f"[~] Backing off {delay:.1f}s before attempt {attempt}.")
            time.sleep(delay)
        result = grader.query(question)
        last_result = result
        content = result.get("content", "")
        if result.get("status") == 200 and is_valid_report(content):
            return result
        if result.get("status") == 200 and is_boilerplate_report(content):
            _log(f"[!] {url} produced boilerplate text; returning it without another call.")
            return result
        if result.get("status") in (413, 424, 503):
            _log(
                f"[!] Not retrying non-transient evaluation failure for {url} "
                f"(status {result.get('status')})."
            )
            return result
        _log(
            f"[!] Attempt {attempt}/{max_retries} produced no valid report for {url} "
            f"(status {result.get('status')})."
        )

    return {
        **last_result,
        "status": last_result.get("status", 500) if last_result.get("status") != 200 else 500,
        "content": (
            f"Evaluation failed for {url} after {max_retries} attempts. "
            f"Last model output: {str(last_result.get('content', ''))[:400] or 'none'}"
        ),
    }
