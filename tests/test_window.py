"""Offline checks that every graded prompt fits the model's certified token window.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from ollama import ResponseError

from contracts import MODEL_WINDOW_TOKENS, PROMPT_TOKEN_BUDGET, RESPONSE_TOKENS
from grading.evaluation import evaluate_batch
from grading.window import ModelWindowError, overflow_tokens, refit_view

URL = "https://github.com/owner/repo"
REPORT = (
    "1. AI Involvement Score (0-100%): 72%\n"
    "Ranked AI evidence (strongest to weakest):\n"
    "Rank 1: `keys.json` holds generated configuration values, which is not decisive alone.\n\n"
    "2. Code Quality Score (0-100%): 81%\n"
    "- Deductions: -15% no test files\n\n"
    "3. Summary & Notable Findings: keys.json holds generated configuration values.\n"
)


def _dense_dump() -> str:
    """Build one evidence view of a file whose tokens are far denser than its characters."""
    header = (
        "REPOSITORY: owner/repo\nFILES FOUND: 1\nFILES FETCHED: 1\n"
        "FILES INCLUDED (1): keys.json\nFILES TRUNCATED: no\n"
    )
    key = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2Nzg5"
    body = "".join(f'  "client-{index:04d}": "{key}{index:04d}",\n' for index in range(40))
    return f"{header}\n--- FILE: keys.json ---\n{body}"


def _overflow(counted: int) -> str:
    """Render the context-size rejection Ollama returns for an oversized prompt."""
    return json.dumps({
        "error": {
            "code": 400,
            "message": (
                f"request ({counted} tokens) exceeds the available context size "
                "(4096 tokens), try increasing it"
            ),
            "type": "exceed_context_size_error",
            "n_prompt_tokens": counted,
            "n_ctx": 4096,
        }
    })


class _WindowGrader:
    """Report configured prompt token counts so window fitting can be exercised offline."""

    def __init__(self, counts: list[int], *, reject_first: str | None = None) -> None:
        """Set the token count each counted prompt reports and an optional first rejection."""
        self.counts = counts
        self.reject_first = reject_first
        self.counted: list[str] = []
        self.sent: list[str] = []

    def count_prompt_tokens(self, messages: list[dict[str, str]]) -> int:
        """Record one counted prompt and report the token count configured for it."""
        self.counted.append(messages[-1]["content"])
        if self.reject_first is not None and len(self.counted) == 1:
            raise ResponseError(self.reject_first, 400)
        index = min(len(self.counted), len(self.counts)) - 1
        return self.counts[index]

    def _chat(self, messages: list[dict[str, str]], seed: int) -> SimpleNamespace:
        """Record one answered prompt and return a valid three-part report."""
        self.sent.append(messages[-1]["content"])
        return SimpleNamespace(
            message=SimpleNamespace(content=REPORT), prompt_eval_count=PROMPT_TOKEN_BUDGET
        )


class TestEvidenceWindowFit(unittest.TestCase):
    """Keep every graded prompt inside the window the model was certified on."""

    def setUp(self) -> None:
        """Provide the repository-wide coverage counts a batch analysis is merged with."""
        self.analysis = {
            "coverage": {
                "files_shown": 1,
                "files_found": 1,
                "files_fetched": 1,
                "truncated": False,
            }
        }

    def _evaluate(self, grader: _WindowGrader) -> dict:
        """Grade one dense view of a repository with a fabricated grader."""
        return evaluate_batch(
            grader, "grade it", URL, _dense_dump(), self.analysis, "owner/repo", 0, 1
        )

    def test_a_prompt_over_the_budget_is_recut_before_the_model_answers(self) -> None:
        """Count the prompt first and re-cut the view before spending a generation on it."""
        grader = _WindowGrader([PROMPT_TOKEN_BUDGET * 2, PROMPT_TOKEN_BUDGET])

        result = self._evaluate(grader)

        self.assertEqual(len(grader.counted), 2)
        self.assertEqual(len(grader.sent), 1)
        self.assertLess(len(grader.counted[1]), len(grader.counted[0]))
        self.assertTrue(result["parsed"]["valid"])
        # The re-cut view no longer reproduces the file the report cites, so that citation
        # is unsupported rather than accepted on the strength of a header line.
        self.assertIn("keys.json", result["unsupported_citations"][0])

    def test_a_prompt_exactly_at_the_budget_is_answered_without_a_refit(self) -> None:
        """Accept a prompt that spends the budget exactly and answer it unchanged."""
        grader = _WindowGrader([PROMPT_TOKEN_BUDGET])

        result = self._evaluate(grader)

        self.assertEqual(grader.sent, grader.counted)
        self.assertTrue(result["parsed"]["valid"])

    def test_a_context_size_rejection_is_retried_with_a_smaller_view(self) -> None:
        """Re-cut a view that Ollama rejected for exceeding its requested context."""
        grader = _WindowGrader(
            [PROMPT_TOKEN_BUDGET, PROMPT_TOKEN_BUDGET],
            reject_first=_overflow(PROMPT_TOKEN_BUDGET * 3),
        )

        result = self._evaluate(grader)

        self.assertEqual(len(grader.counted), 2)
        self.assertEqual(len(grader.sent), 1)
        self.assertLess(len(grader.counted[1]), len(grader.counted[0]))
        self.assertTrue(result["parsed"]["valid"])

    def test_a_view_that_cannot_be_fitted_is_never_sent_to_the_model(self) -> None:
        """Refuse the batch rather than asking the model to read a prompt it cannot hold."""
        grader = _WindowGrader([PROMPT_TOKEN_BUDGET * 4] * 8)

        with self.assertRaises(ModelWindowError):
            self._evaluate(grader)

        self.assertEqual(grader.sent, [])

    def test_the_refit_cuts_the_characters_its_overflow_accounts_for(self) -> None:
        """Cut the view by what the measured overflow costs, not by a fixed fraction."""
        view = _dense_dump()
        excess = PROMPT_TOKEN_BUDGET

        cut = refit_view(view, required=PROMPT_TOKEN_BUDGET * 2, prompt_chars=len(view))

        removed = len(view) - len(cut)
        self.assertGreaterEqual(removed, excess)
        self.assertLessEqual(removed, int(excess * 1.3))

    def test_the_response_reserve_stays_inside_the_trained_window(self) -> None:
        """Keep the prompt budget and the generation reserve inside the certified window."""
        self.assertGreater(PROMPT_TOKEN_BUDGET, 0)
        self.assertGreater(RESPONSE_TOKENS, 0)
        self.assertEqual(PROMPT_TOKEN_BUDGET + RESPONSE_TOKENS, MODEL_WINDOW_TOKENS)


class TestOverflowReporting(unittest.TestCase):
    """Read the token counts out of every shape a context-size rejection can take."""

    def test_a_json_rejection_reports_its_prompt_and_context_tokens(self) -> None:
        """Parse the JSON body Ollama returns when a prompt exceeds the context."""
        error = ResponseError(_overflow(4286), 400)
        self.assertEqual(overflow_tokens(error), (4286, 4096))

    def test_a_python_repr_rejection_reports_its_prompt_and_context_tokens(self) -> None:
        """Parse the mapping repr the client stringifies when it unwraps the error body."""
        error = ResponseError(repr({"n_prompt_tokens": 5220, "n_ctx": 4096}), 400)
        self.assertEqual(overflow_tokens(error), (5220, 4096))

    def test_an_unrelated_rejection_is_not_read_as_an_overflow(self) -> None:
        """Leave other server rejections to the retryable failure path."""
        error = ResponseError(json.dumps({"error": {"message": "model is restarting"}}), 500)
        self.assertIsNone(overflow_tokens(error))


if __name__ == "__main__":
    unittest.main()
