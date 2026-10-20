"""Evaluate evidence views, seed them deterministically, and aggregate reports.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import hashlib
from typing import Any

from ollama import ResponseError

from contracts import (
    PROMPT_TOKEN_BUDGET,
    SYSTEM_PROMPT,
    batch_question,
    build_user_message,
    combine_reports,
    parse_report,
    source_line_count,
    verify_citations,
)
from contracts.prompts import FILE_NAME_HEADER
from detection import prompt_context

from .ollama import _log
from .window import (
    MAX_WINDOW_ATTEMPTS,
    MIN_VIEW_CHARS,
    ModelWindowError,
    overflow_tokens,
    refit_view,
)


def seed_for(key: str, batch_index: int = 0) -> int:
    """Create stable per-repository seeds that vary across evidence batches."""
    material = f"{key}:{batch_index}"
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:8], 16)


def service_failure(host: str, model: str, url: str, detail: str) -> dict[str, Any]:
    """Return an actionable availability error without fetching repository data."""
    return {
        "status": 503,
        "content": (
            f"Ollama is unavailable at {host}: {detail}. Confirm OLLAMA_HOST and "
            f"that '{model}' is installed on that host."
        ),
        "analysis": None,
        "url": url,
    }


def _batch_analysis(view: str, repository_analysis: dict[str, Any]) -> dict[str, Any]:
    """Measure one evidence view and attach repository-wide coverage counts."""
    batch_analysis = prompt_context(view)
    batch_analysis["coverage"] = {
        **batch_analysis["coverage"],
        "files_found": repository_analysis["coverage"].get("files_found", 0),
        "files_fetched": repository_analysis["coverage"].get("files_fetched", 0),
        "truncated": repository_analysis["coverage"].get("truncated", False),
    }
    return batch_analysis


def _messages(
    question: str,
    url: str,
    view: str,
    batch_analysis: dict[str, Any],
    batch_index: int,
    batch_count: int,
) -> list[dict[str, str]]:
    """Assemble one grading conversation around a single evidence view."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_message(
                batch_question(question, batch_index + 1, batch_count),
                view,
                url=url,
                analysis=batch_analysis,
            ),
        },
    ]


def _prompt_chars(messages: list[dict[str, str]]) -> int:
    """Return the characters the whole conversation spends, system prompt included."""
    return len(SYSTEM_PROMPT) + len(messages[-1]["content"])


def _refit(
    view: str,
    attempt: int,
    required: int,
    prompt_chars: int,
    batch_index: int,
    batch_count: int,
) -> str:
    """Shrink a view that needs more prompt tokens than the model window allows."""
    if attempt + 1 >= MAX_WINDOW_ATTEMPTS or len(view) <= MIN_VIEW_CHARS:
        raise ModelWindowError(
            f"evidence batch {batch_index + 1}/{batch_count} needs {required} prompt "
            f"token(s) and cannot be fitted to the {PROMPT_TOKEN_BUDGET}-token prompt budget"
        )
    _log(
        f"[i] refitting evidence batch {batch_index + 1}/{batch_count}: {required} prompt "
        f"token(s) against the {PROMPT_TOKEN_BUDGET}-token budget"
    )
    return refit_view(view, required, prompt_chars)


def _send_view(
    grader: Any,
    question: str,
    url: str,
    view: str,
    repository_analysis: dict[str, Any],
    seed: int,
    batch_index: int,
    batch_count: int,
) -> tuple[Any, str]:
    """Send one view, re-cutting it until the whole prompt fits the model window."""
    for attempt in range(MAX_WINDOW_ATTEMPTS):
        analysis = _batch_analysis(view, repository_analysis)
        messages = _messages(question, url, view, analysis, batch_index, batch_count)
        try:
            required = grader.count_prompt_tokens(messages)
            if required <= PROMPT_TOKEN_BUDGET:
                return grader._chat(messages, seed), view
        except ResponseError as exc:
            reported = overflow_tokens(exc)
            if reported is None:
                raise
            required = reported[0]
        view = _refit(
            view, attempt, required, _prompt_chars(messages), batch_index, batch_count
        )
    raise ModelWindowError(
        f"evidence batch {batch_index + 1}/{batch_count} could not be fitted to the "
        f"{PROMPT_TOKEN_BUDGET}-token prompt budget"
    )


def evaluate_batch(
    grader: Any,
    question: str,
    url: str,
    batch: str,
    repository_analysis: dict[str, Any],
    seed_key: str,
    batch_index: int,
    batch_count: int,
) -> dict[str, Any]:
    """Evaluate one batch and verify citations against precisely that evidence view."""
    response, view = _send_view(
        grader,
        question,
        url,
        batch,
        repository_analysis,
        seed_for(seed_key, batch_index),
        batch_index,
        batch_count,
    )
    content = (response.message.content or "").strip()
    return {
        "content": content,
        "parsed": parse_report(content),
        "unsupported_citations": verify_citations(content, view),
        "lines": source_line_count(view),
    }


def _single_batch(result: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    """Return one report only when its individual response passed validation."""
    if not result["parsed"]["valid"]:
        return {
            "status": 502,
            "content": f"The grader returned an invalid report: {result['content'][:400]}",
            "analysis": analysis,
            "unsupported_citations": result["unsupported_citations"],
        }
    return {
        "status": 200,
        "content": result["content"],
        "analysis": analysis,
        "unsupported_citations": result["unsupported_citations"],
    }


def _add_repository_context(
    aggregate: dict[str, Any], full_dump: str
) -> dict[str, Any]:
    """Append repository-level file count without changing the report structure."""
    if aggregate["status"] == 200:
        names = list(dict.fromkeys(FILE_NAME_HEADER.findall(full_dump)))
        if names:
            summary = "\n\n3. Summary"
            suffix = f" The reviewed repository included {len(names)} fetched file(s)."
            aggregate["content"] = aggregate["content"].replace(summary, summary + suffix)
    return aggregate


def combine_batches(
    results: list[dict[str, Any]],
    full_dump: str,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate valid reports and preserve repository-wide coverage metadata."""
    if len(results) == 1:
        return _single_batch(results[0], analysis)
    aggregate = combine_reports(results, [result["lines"] for result in results])
    aggregate["analysis"] = analysis
    return _add_repository_context(aggregate, full_dump)
