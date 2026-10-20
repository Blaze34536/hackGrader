"""Configure the Ollama client and check host and model readiness.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ollama import Client

from contracts import MODEL_WINDOW_TOKENS, RESPONSE_TOKENS

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
_DEFAULT_MODEL = "hackgrader"
_RELEASED_MODEL = (
    Path(__file__).resolve().parents[2] / "model" / "data" / "deployed_model.json"
)


def released_model_tag() -> str | None:
    """Return the tag the model pipeline last verified and registered, if any."""
    try:
        record = json.loads(_RELEASED_MODEL.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    tag = record.get("tag")
    return tag.strip() if isinstance(tag, str) and tag.strip() else None


def _log(message: str) -> None:
    """Print a log message even when the terminal encoding is ASCII-only."""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", "replace").decode("ascii"))


def prompt_tokens(response: Any) -> int:
    """Read the prompt token count one completed request reported."""
    counted = getattr(response, "prompt_eval_count", None)
    if counted is None and isinstance(response, dict):
        counted = response.get("prompt_eval_count")
    try:
        return max(0, int(counted))
    except (TypeError, ValueError):
        return 0


def _normalise_host(host: str) -> str:
    """Normalize OLLAMA_HOST values, including bare host:port values."""
    candidate = host.strip().rstrip("/")
    if candidate and "://" not in candidate:
        candidate = f"http://{candidate}"
    return candidate or DEFAULT_OLLAMA_HOST


class OllamaClient:
    """Manage the selected Ollama host, model tag, and chat client."""

    def __init__(
        self,
        target_model: str | None = None,
        host: str | None = None,
        chat_timeout: float = 240.0,
        readiness_timeout: float = 5.0,
        batch_budget: int | None = None,
        *,
        _client_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Configure Ollama host, model, readiness timeouts, and clients."""
        target_model = (
            target_model
            or os.environ.get("HACKGRADER_MODEL")
            or released_model_tag()
            or _DEFAULT_MODEL
        )
        resolved_host = host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
        self.model = target_model
        self.ollama_host = _normalise_host(resolved_host)
        self.readiness_timeout = max(0.1, readiness_timeout)
        self.batch_budget = batch_budget
        self._model_list_error: str | None = None
        self._last_ollama_error: str | None = None
        client_factory = _client_factory or Client
        self.client = client_factory(host=self.ollama_host, timeout=chat_timeout)
        self._list_client = client_factory(
            host=self.ollama_host, timeout=min(chat_timeout, 5.0)
        )

        installed_models = self._installed_models()
        aliases = {target_model, f"{target_model}:latest"}
        configured = next((name for name in installed_models if name in aliases), None)
        if configured:
            self.model = configured
        elif installed_models:
            _log(
                f"[!] '{target_model}' is not installed at {self.ollama_host}; "
                "the configured grader model will not be silently replaced. Set "
                "HACKGRADER_MODEL to an installed tag."
            )
        elif self._model_list_error:
            _log(
                f"[!] Could not list Ollama models at {self.ollama_host}: "
                f"{self._model_list_error}"
            )
        else:
            _log(
                f"[!] No models are installed at {self.ollama_host}. "
                f"Run `ollama pull {target_model}` or set HACKGRADER_MODEL."
            )
        _log(f"[i] Ollama host: {self.ollama_host}")
        _log(f"[i] Grader model: {self.model}")

    def _installed_models(self) -> list[str]:
        """Return names served at the configured host, keeping API errors for diagnostics."""
        try:
            response = self._list_client.list()
            if hasattr(response, "models"):
                return [entry.model for entry in response.models]
            if isinstance(response, dict):
                return [entry.get("model", "") for entry in response.get("models", [])]
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._model_list_error = f"{type(exc).__name__}: {exc}"
        return []

    def wait_for_ollama(
        self, timeout: float | None = None, poll_interval: float = 0.25
    ) -> bool:
        """Probe the configured host until ready or until the timeout expires."""
        timeout = self.readiness_timeout if timeout is None else max(0.0, timeout)
        deadline = time.monotonic() + timeout
        while True:
            try:
                request = urllib.request.Request(f"{self.ollama_host}/api/version")
                with urllib.request.urlopen(
                    request, timeout=min(1.5, max(timeout, 0.1))
                ) as response:
                    if response.status == 200:
                        self._last_ollama_error = None
                        return True
                    self._last_ollama_error = (
                        f"readiness endpoint returned HTTP {response.status}"
                    )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                reason = getattr(exc, "reason", exc)
                self._last_ollama_error = f"{type(reason).__name__}: {reason}"
            except (RuntimeError, TypeError, ValueError) as exc:
                self._last_ollama_error = f"{type(exc).__name__}: {exc}"

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(poll_interval, remaining))

    def _available_model(self) -> bool:
        """Verify the selected Ollama tag, allowing only `:latest` alias equivalence."""
        installed = self._installed_models()
        aliases = {self.model, self.model.removesuffix(":latest")}
        if any(
            name in aliases or name.removesuffix(":latest") in aliases
            for name in installed
        ):
            return True
        self._last_ollama_error = (
            f"model '{self.model}' is not listed at {self.ollama_host}"
            + (f"; list failed: {self._model_list_error}" if self._model_list_error else "")
        )
        return False

    def preflight_error(self) -> str | None:
        """Describe host/model unavailability before a gallery run starts."""
        if not self.wait_for_ollama():
            return (
                f"Ollama did not answer at {self.ollama_host}/api/version: "
                f"{self._last_ollama_error or 'readiness probe timed out'}. "
                "Set OLLAMA_HOST to the running Ollama server."
            )
        if not self._available_model():
            return (
                f"The configured grader model '{self.model}' is unavailable at "
                f"{self.ollama_host}: {self._last_ollama_error}. "
                "Set HACKGRADER_MODEL or install the model on that Ollama server."
            )
        return None

    def _options(self, **extra: Any) -> dict[str, Any]:
        """Return the decoding options every request shares.

        The context is the larger of the requested length and the window the model was
        certified on, so a small ``HACKGRADER_OLLAMA_NUM_CTX`` can never make the
        certified evidence budget impossible to serve. ``num_predict`` is the same
        response reserve the prompt budget subtracts, so the two cannot disagree.
        """
        requested = int(os.environ.get("HACKGRADER_OLLAMA_NUM_CTX", "4096"))
        return {
            "num_ctx": max(requested, MODEL_WINDOW_TOKENS),
            "temperature": 0.2,
            "num_predict": RESPONSE_TOKENS,
            **extra,
        }

    def count_prompt_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count one prompt with the model's own tokenizer before asking it to answer.

        Ollama cannot count a prompt without reading it, so this spends one generated
        token instead of a whole report; a prompt that exceeds the context is reported
        through the same rejection the answering call would raise.
        """
        response = self.client.chat(
            model=self.model, messages=messages, options=self._options(num_predict=1)
        )
        return prompt_tokens(response)

    def _chat(self, messages: list[dict[str, Any]], seed: int) -> Any:
        """Send one completion request with stable decoding and the certified window."""
        return self.client.chat(
            model=self.model, messages=messages, options=self._options(seed=seed)
        )
