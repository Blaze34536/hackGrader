"""Fit each evidence view inside the token window the grader model was trained on.

Training renders every candidate row and re-cuts the repository evidence until the
tokenized conversation fits ``HACKGRADER_MAX_LEN``. Serving used to bound only the
evidence view, in characters, with no bound on the whole prompt: the shared system
prompt, the measurement block, and the question were all charged to the same context
after the fact. A repository whose files tokenize densely - an encoded asset, a
generated lockfile - then produced a prompt larger than both the trained window and the
requested context, and the request failed outright.

Every correction here is measured, not estimated: the serving path counts the prompt
with the model's own tokenizer before generating, and cuts the view by the characters
that overflow accounts for.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import re

from contracts import PROMPT_TOKEN_BUDGET, trim_dump

MIN_VIEW_CHARS = 256
MAX_WINDOW_ATTEMPTS = 4
REFIT_MARGIN = 1.15

_PROMPT_TOKENS = re.compile(r"['\"]?n_prompt_tokens['\"]?\s*[:=]\s*(\d+)")
_CONTEXT_TOKENS = re.compile(r"['\"]?n_ctx['\"]?\s*[:=]\s*(\d+)")


class ModelWindowError(RuntimeError):
    """Report that an evidence view cannot be fitted inside the model window."""


def overflow_tokens(error: BaseException) -> tuple[int, int] | None:
    """Read the reported prompt and context token counts from a context-size rejection."""
    text = f"{getattr(error, 'error', '')} {error}"
    counted = _PROMPT_TOKENS.search(text)
    context = _CONTEXT_TOKENS.search(text)
    if not counted or not context:
        return None
    return int(counted.group(1)), int(context.group(1))


def refit_view(view: str, required: int, prompt_chars: int) -> str:
    """Cut the view by the characters its measured token overflow accounts for."""
    excess = max(1, required - PROMPT_TOKEN_BUDGET)
    chars_per_token = max(1.0, prompt_chars / max(1, required))
    target = len(view) - int(excess * chars_per_token * REFIT_MARGIN)
    return trim_dump(view, budget=max(MIN_VIEW_CHARS, target))
