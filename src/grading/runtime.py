"""Fetch repositories and coordinate bounded, validated Ollama evaluations.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import re
from typing import Any

import aiohttp
from ollama import ResponseError

from contracts import (
    BASE_QUESTION,
    PROMPT_TOKEN_BUDGET,
    REPORT_INSTRUCTION,
    SYSTEM_PROMPT,
    batch_question,
    build_user_message,
    parse_report,
    split_dump,
)
from detection import prompt_context
from repositories.reader import read_repository
from repositories.urls import is_github_url, repo_slug

from .evaluation import combine_batches, evaluate_batch, service_failure
from .ollama import OllamaClient, _log
from .retry import query_with_retry as _query_with_retry
from .window import ModelWindowError

__all__ = [
    "BASE_QUESTION",
    "REPORT_INSTRUCTION",
    "SYSTEM_PROMPT",
    "Model",
    "batch_question",
    "build_user_message",
    "extract_url",
]

URL_PATTERN = re.compile(r"https?://[^\s<>\"')]+")


def extract_url(text: str) -> str:
    """Return the first GitHub URL in a question, or the first URL otherwise."""
    matches = [match.rstrip(".,;:") for match in URL_PATTERN.findall(text or "")]
    for match in matches:
        if is_github_url(match):
            return match
    return matches[0] if matches else ""


class Model(OllamaClient):
    """Fetch repository data and produce validated Ollama grader reports."""

    @staticmethod
    def extract_url(question: str) -> str:
        """Return the repository URL named in a question.

        The retry wrapper names the repository in its status messages through the
        grader it is given, so the model exposes this alongside ``query``.
        """
        return extract_url(question)

    @staticmethod
    def _read_repository(url: str) -> str:
        """Read a GitHub repository with the shared bounded content reader."""
        return read_repository(url)

    def query(self, question: str) -> dict[str, Any]:
        """Fetch and evaluate a repository in bounded, complete-view batches."""
        url = extract_url(question)
        if not url:
            return {
                "status": 424,
                "content": "Could not read the repository. No repository URL was found in the request.",
                "analysis": None,
            }
        if not is_github_url(url):
            return {
                "status": 424,
                "content": f"Could not read {url}. Only GitHub repository URLs are supported.",
                "analysis": None,
            }
        if not self.wait_for_ollama():
            return service_failure(
                self.ollama_host, self.model, url,
                self._last_ollama_error or "readiness probe timed out",
            )
        if not self._available_model():
            return service_failure(
                self.ollama_host, self.model, url,
                self._last_ollama_error or "configured model is missing",
            )

        repo_block = self._read_repository(url)
        if not repo_block or repo_block.startswith(("Error", "Repository ")):
            _log(f"[!] Could not read {url}: {repo_block[:160]}")
            return {
                "status": 424,
                "content": f"Could not read {url}. {repo_block or 'No repository content was returned.'}",
                "analysis": None,
            }

        analysis = prompt_context(repo_block)
        try:
            batches = split_dump(repo_block, budget=self.batch_budget)
        except ValueError as exc:
            return {
                "status": 413,
                "content": f"The repository cannot fit the configured model batch budget: {exc}",
                "analysis": analysis,
            }
        if not batches:
            return {
                "status": 424,
                "content": f"Could not read {url}. The repository dump was empty.",
                "analysis": analysis,
            }

        seed_key = repo_slug(url)
        _log(
            f"[i] judge will review {len(batches)} batch(es), "
            f"covering {analysis['coverage']['files_shown']} fetched file(s)"
        )
        results = []
        for index, batch in enumerate(batches):
            try:
                results.append(
                    evaluate_batch(
                        self, question, url, batch, analysis, seed_key, index, len(batches)
                    )
                )
            except ModelWindowError as exc:
                _log(f"[!] {url}: batch {index + 1}/{len(batches)} does not fit the window")
                return {
                    "status": 413,
                    "content": (
                        f"Repository batch {index + 1}/{len(batches)} cannot be graded inside "
                        f"the {PROMPT_TOKEN_BUDGET}-token prompt budget the model was trained "
                        f"on: {exc}"
                    ),
                    "analysis": analysis,
                    "batch_count": len(batches),
                    "batch_reports": [result["content"] for result in results],
                }
            except ResponseError as exc:
                _log(f"[!] {url}: Ollama refused batch {index + 1}/{len(batches)}: {exc}")
                return {
                    "status": 502,
                    "content": (
                        f"Ollama refused repository batch {index + 1}/{len(batches)}: "
                        f"{getattr(exc, 'error', exc)}"
                    ),
                    "analysis": analysis,
                    "batch_count": len(batches),
                    "batch_reports": [result["content"] for result in results],
                }
            except (
                aiohttp.ClientError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                return {
                    "status": 502,
                    "content": (
                        f"Error during chat for repository batch {index + 1}/{len(batches)}: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "analysis": analysis,
                    "batch_count": len(batches),
                    "batch_reports": [result["content"] for result in results],
                }

        # Aggregation keeps the judge's ranked evidence from valid batches and drops
        # citations from reports that did not validate.
        combined = combine_batches(results, repo_block, analysis)
        combined["batch_count"] = len(batches)
        combined["batch_reports"] = [result["content"] for result in results]
        if combined.get("failed_batches"):
            _log(
                f"[!] {url}: {len(combined['failed_batches'])} of {len(batches)} evidence "
                "batch(es) produced no valid report; aggregating the batches that did"
            )
        if combined["status"] != 200:
            return combined
        parsed = parse_report(combined["content"])
        _log(
            f"[i] judge: AI {parsed['ai_score']}% / quality {parsed['code_score']}%"
        )
        return combined

    def query_with_retry(
        self, question: str, max_retries: int = 3, initial_delay: float = 3.0
    ) -> dict[str, Any]:
        """Retry transient errors while returning input and configuration failures promptly."""
        return _query_with_retry(self, question, max_retries, initial_delay)
