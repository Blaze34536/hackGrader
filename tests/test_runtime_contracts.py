"""Offline regressions for Ollama readiness and repository evidence batches.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

from ollama import ResponseError

import grading.runtime as grader_runtime
from contracts import (
    PROMPT_TOKEN_BUDGET,
    REPORT_INSTRUCTION,
    SYSTEM_PROMPT,
    authorship_evidence,
    batch_question,
    combine_reports,
    evidence_claims,
    extract_evidence,
    measurement_block,
    parse_report,
    source_line_count,
    split_dump,
    verify_citations,
)
from contracts.prompts import FILE_NAME_HEADER
from contracts.repository_dump import _parse_metadata
from grading.reporting import Reporting
from repositories.collection import collect_repository
from repositories.discovery import walk_directories
from repositories.selection import (
    is_candidate,
    is_documentation,
    looks_binary,
    redact_payloads,
    selection_priority,
)
from repositories.settings import MAX_DOC_CHARS
from repositories.transport import make_fetcher

DENSE_DUMP = (
    "REPOSITORY: owner/repo\nFILES FOUND: 1\nFILES FETCHED: 1\n"
    "FILES INCLUDED (1): keys.json\nFILES TRUNCATED: no\n\n"
    "--- FILE: keys.json ---\n"
    + "".join(f'  "client-{index:04d}": "dGhpcyBpcyBnZW5lcmF0ZWQgZGF0YQ==",\n' for index in range(120))
)
REPO_QUESTION = "grade https://github.com/owner/repo"
EVIDENCE_URL = "https://github.com/owner/repo"

# One repository split into three views where the last one comes back without a report.
PARTIAL_VIEWS = (
    "REPOSITORY: owner/repo\nFILES FOUND: 2\nFILES FETCHED: 2\n"
    "FILES INCLUDED (1): app.py\nFILES TRUNCATED: no\n\n--- FILE: app.py ---\nprint('a')\n",
    "REPOSITORY: owner/repo\nFILES FOUND: 2\nFILES FETCHED: 2\n"
    "FILES INCLUDED (1): util.py\nFILES TRUNCATED: no\n\n--- FILE: util.py ---\nprint('b')\n",
    "REPOSITORY: owner/repo\nFILES FOUND: 2\nFILES FETCHED: 2\n"
    "FILES INCLUDED (1): keys.json\nFILES TRUNCATED: no\n\n--- FILE: keys.json ---\n{}\n",
)
VALID_REPORT_A = (
    "1. AI Involvement Score (0-100%): 70%\n"
    "Ranked AI evidence (strongest to weakest):\n"
    "Rank 1: `print('a')` in app.py is the only statement in the file\n\n"
    "2. Code Quality Score (0-100%): 60%\n"
    "- Deductions: -10% no tests present\n\n"
    "3. Summary & Notable Findings: A short program that prints one value.\n"
)
VALID_REPORT_B = (
    "1. AI Involvement Score (0-100%): 80%\n"
    "Ranked AI evidence (strongest to weakest):\n"
    "Rank 1: `print('b')` in util.py repeats one statement twice\n\n"
    "2. Code Quality Score (0-100%): 50%\n"
    "- Deductions: -20% duplicated logic\n\n"
    "3. Summary & Notable Findings: A utility module with repeated branches.\n"
)
INVALID_REPORT = "the model described the project without ever giving a score"
BINARY_BODY = "PK\x03\x04\x00\x00model_state_dictq\x02\x00Rr" + "\x10" * 60


class _ListClient:
    """Provide a configurable Ollama model list without an HTTP request."""

    def __init__(self, models: list[str] | None = None, error: Exception | None = None) -> None:
        """Set the fake installed model names and optional list failure."""
        self.models = models or []
        self.error = error

    def list(self):
        """Return configured fake Ollama model metadata or raise its failure."""
        if self.error:
            raise self.error
        return SimpleNamespace(models=[SimpleNamespace(model=name) for name in self.models])


class TestRuntimeConfiguration(unittest.TestCase):
    """Check host/tag configuration and actionable readiness handling."""

    def _model(self, models: list[str] | None = None) -> grader_runtime.Model:
        """Create an Ollama grader that reports only the supplied fake model tags."""
        with patch("grading.ollama.Client") as client_factory:
            client_factory.return_value = _ListClient(models)
            grader = grader_runtime.Model(readiness_timeout=0.1)
        grader._list_client = _ListClient(models)
        return grader

    def test_host_environment_is_used_without_silently_changing_tags(self) -> None:
        """Honor configured Ollama hosts and preserve exact model tag requests."""
        with (
            patch.dict(
                "os.environ",
                {"OLLAMA_HOST": "grader.example:11434", "HACKGRADER_MODEL": "hackgrader:build"},
            ),
            patch("grading.ollama.Client") as client_factory,
        ):
            client_factory.return_value = _ListClient(["other:latest"])
            grader = grader_runtime.Model(readiness_timeout=0.1)

        self.assertEqual(grader.ollama_host, "http://grader.example:11434")
        self.assertEqual(grader.model, "hackgrader:build")
        self.assertEqual(client_factory.call_args.kwargs["host"], "http://grader.example:11434")

    def test_released_model_tag_is_adopted_when_the_environment_is_unset(self) -> None:
        """Use the build the model pipeline verified when no tag is configured."""
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            pointer = Path(directory) / "deployed_model.json"
            pointer.write_text(json.dumps({"tag": "hackgrader:cafe"}), encoding="utf-8")
            with (
                patch("grading.ollama._RELEASED_MODEL", pointer),
                patch.dict("os.environ", {}, clear=True),
                patch("grading.ollama.Client") as client_factory,
            ):
                client_factory.return_value = _ListClient(["hackgrader:cafe"])
                grader = grader_runtime.Model(readiness_timeout=0.1)

        self.assertEqual(grader.model, "hackgrader:cafe")

    def test_released_tag_missing_from_ollama_is_not_substituted(self) -> None:
        """Report a missing released build instead of grading with another model."""
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            pointer = Path(directory) / "deployed_model.json"
            pointer.write_text(json.dumps({"tag": "hackgrader:cafe"}), encoding="utf-8")
            with (
                patch("grading.ollama._RELEASED_MODEL", pointer),
                patch.dict("os.environ", {}, clear=True),
                patch("grading.ollama.Client") as client_factory,
            ):
                client_factory.return_value = _ListClient(["other:latest"])
                grader = grader_runtime.Model(readiness_timeout=0.1)

        self.assertEqual(grader.model, "hackgrader:cafe")
        self.assertFalse(grader._available_model())

    def test_latest_alias_is_accepted_but_unrelated_model_is_not(self) -> None:
        """Accept the unqualified latest model tag without accepting other builds."""
        grader = self._model(["hackgrader:latest"])
        grader.model = "hackgrader"
        self.assertTrue(grader._available_model())
        grader.model = "hackgrader:new-build"
        self.assertFalse(grader._available_model())

    def test_preflight_returns_a_readiness_diagnostic(self) -> None:
        """Explain unreachable Ollama hosts before any repository evaluation starts."""
        grader = self._model(["hackgrader:latest"])
        with patch.object(grader, "wait_for_ollama", return_value=False):
            grader._last_ollama_error = "ConnectionRefusedError: refused"
            error = grader.preflight_error()
        self.assertIsNotNone(error)
        self.assertIn("ConnectionRefusedError", error)
        self.assertIn("OLLAMA_HOST", error)

    def test_empty_url_is_rejected_before_readiness_or_fetch(self) -> None:
        """Reject requests without a repository URL before network activity."""
        grader = self._model(["hackgrader:latest"])
        with (
            patch.object(grader, "wait_for_ollama") as ready,
            patch("grading.runtime.read_repository") as read_repo,
        ):
            result = grader.query("grade this project")
        self.assertEqual(result["status"], 424)
        ready.assert_not_called()
        read_repo.assert_not_called()


class TestSharedBatchContract(unittest.TestCase):
    """Ensure each serving batch carries truthful view coverage and stable training text."""

    def test_split_views_mark_fetched_files_not_included_in_each_view(self) -> None:
        """Identify other fetched paths on every bounded model evidence view."""
        dump = (
            "REPOSITORY: owner/repo\nDEFAULT BRANCH: main\n"
            "FILES FOUND: 3\nFILES FETCHED: 3\n"
            "FILES INCLUDED (3): README.md, app.py, util.py\n"
            "FILES TRUNCATED: no\n\n"
            "--- FILE: README.md ---\n" + "readme\n" * 12
            + "\n\n--- FILE: app.py ---\n" + "print('app')\n" * 12
            + "\n\n--- FILE: util.py ---\n" + "print('util')\n" * 12
        )
        batches = split_dump(dump, budget=550)

        # Several files share a view now, so a view can hold all three. The contract is that a
        # view which leaves a fetched file out names the omission, and that every file is shown
        # somewhere across the views.
        fetched = {"README.md", "app.py", "util.py"}
        shown_somewhere: set[str] = set()
        self.assertGreater(len(batches), 1)
        for batch in batches:
            self.assertLessEqual(len(batch), 550)
            shown = {name.strip() for name in FILE_NAME_HEADER.findall(batch)}
            self.assertTrue(shown)
            shown_somewhere |= shown
            if shown != fetched:
                self.assertRegex(batch, r"(?m)^FILES NOT INCLUDED \(\d+\):")
        self.assertEqual(shown_somewhere, fetched)

    def test_measurements_separate_repository_fetches_from_current_view(self) -> None:
        """Keep discovery, fetch, and current-batch coverage counts distinct."""
        dump = (
            "REPOSITORY: owner/repo\nFILES FOUND: 4\nFILES FETCHED: 3\n"
            "FILES INCLUDED (1): main.py\nFILES TRUNCATED: yes\n"
            "FILES NOT INCLUDED (2): other.py, util.py\n"
            "FILES NOT FETCHED (1): missing.py\n\n"
            "--- FILE: main.py ---\nprint('x')\n"
        )
        metrics = {
            "source_files": 1, "source_lines": 1, "lines": 1, "chars": len(dump),
            "readme_chars": 0, "readme_headings": 0, "readme_bullets": 0,
            "readme_first_person": 0, "readme_casual": 0, "readme_typos": 0,
            "functions": 0, "docstrings": 0, "restated_docstrings": 0,
            "typed_functions": 0, "jsdoc_blocks": 0, "todos": 0,
            "debug_prints": 0, "commented_out": 0, "bare_excepts": 0,
            "credentials": 0, "localhost": 0, "sql_interpolation": 0,
            "local_paths": 0, "emoji": 0, "code_lines": 1, "comment_lines": 0,
            "spacing_slips": 0, "spacing_sites": 0, "tab_lines": 0,
            "trailing_lines": 0, "python_syntax_errors": 0, "python_files": 1,
            "longest_function": 0, "longest_function_name": "",
        }
        analysis = {
            "metrics": metrics,
            "coverage": {"files_shown": 1, "files_found": 4, "files_fetched": 3, "truncated": True},
        }
        rendered = measurement_block(analysis, dump)

        self.assertIn("4 relevant path(s) found; 3 fetched; 1 reproduced", rendered)
        self.assertIn("fetched but not in this batch: 2", rendered)
        self.assertIn("not fetched: 1", rendered)

    def test_formatting_counts_are_not_shown_to_the_judge(self) -> None:
        """Keep the style counts the judge cited as authorship evidence out of a view.

        The unspaced-comma count flags ordinary style such as
        `import React, { useEffect }`, and the tab count read as machine-like tidiness: a
        project indented with spaces reported zero tab-indented lines, and the judge ranked
        "0 line(s) are tab indented" as its strongest sign. Neither count reaches it now.
        """
        metrics = {
            "source_files": 1, "source_lines": 1, "lines": 1, "chars": 1,
            "readme_chars": 0, "readme_headings": 0, "readme_bullets": 0,
            "readme_first_person": 0, "readme_casual": 0, "readme_typos": 0,
            "functions": 0, "docstrings": 0, "restated_docstrings": 0,
            "typed_functions": 0, "jsdoc_blocks": 0, "todos": 0,
            "debug_prints": 0, "commented_out": 0, "bare_excepts": 0,
            "credentials": 0, "localhost": 0, "sql_interpolation": 0,
            "local_paths": 0, "emoji": 0, "code_lines": 1, "comment_lines": 0,
            "spacing_slips": 19, "spacing_sites": 52, "tab_lines": 2,
            "trailing_lines": 0, "python_syntax_errors": 0, "python_files": 1,
            "longest_function": 0, "longest_function_name": "",
        }
        analysis = {
            "metrics": metrics,
            "coverage": {"files_shown": 1, "files_found": 1, "files_fetched": 1, "truncated": False},
        }
        rendered = measurement_block(analysis)

        self.assertNotIn("Unspaced", rendered)
        self.assertNotIn("19 of 52", rendered)
        self.assertNotIn("Tab-indented", rendered)
        self.assertNotIn("whitespace", rendered)
        self.assertIn("Source code lines: 1", rendered)

    def test_training_and_runtime_use_the_same_batch_question(self) -> None:
        """Ensure runtime and candidate examples use the identical prompt framing."""
        question = "Evaluate https://github.com/owner/repo"
        self.assertIn("repository evidence batch 1 of 2", batch_question(question, 1, 2))
        self.assertIn("cite only those files", grader_runtime.batch_question(question, 1, 2))


class TestWindowOverflowHandling(unittest.TestCase):
    """Ensure one repository that cannot fit the model window fails on its own."""

    def _grader(self) -> grader_runtime.Model:
        """Create an Ollama grader that reports only the released model tag."""
        with patch("grading.ollama.Client") as client_factory:
            client_factory.return_value = _ListClient(["hackgrader:latest"])
            grader = grader_runtime.Model(readiness_timeout=0.1)
        grader._list_client = _ListClient(["hackgrader:latest"])
        return grader

    def test_a_repository_that_cannot_fit_the_window_reports_413(self) -> None:
        """Return a non-retryable status instead of raising out of the gallery run."""
        grader = self._grader()
        with (
            patch.object(grader, "wait_for_ollama", return_value=True),
            patch.object(grader, "_available_model", return_value=True),
            patch("grading.runtime.read_repository", return_value=DENSE_DUMP),
            patch.object(
                grader, "count_prompt_tokens", return_value=PROMPT_TOKEN_BUDGET * 4
            ),
            patch.object(grader, "_chat") as chat,
        ):
            result = grader.query(REPO_QUESTION)

        self.assertEqual(result["status"], 413)
        self.assertIn("prompt budget", result["content"])
        chat.assert_not_called()

    def test_a_rejected_request_returns_a_retryable_status(self) -> None:
        """Turn any other Ollama rejection into a retryable result, not a crash."""
        grader = self._grader()
        rejection = ResponseError(
            json.dumps({"error": {"code": 500, "message": "model is restarting"}}), 500
        )
        with (
            patch.object(grader, "wait_for_ollama", return_value=True),
            patch.object(grader, "_available_model", return_value=True),
            patch("grading.runtime.read_repository", return_value=DENSE_DUMP),
            patch.object(grader, "count_prompt_tokens", side_effect=rejection),
        ):
            result = grader.query(REPO_QUESTION)

        self.assertEqual(result["status"], 502)
        self.assertIn("model is restarting", result["content"])


class _FakeRequest:
    """Serve one canned response as an asynchronous context manager."""

    def __init__(self, response: Any) -> None:
        """Hold the response this request will hand back."""
        self._response = response

    async def __aenter__(self) -> Any:
        """Return the canned response."""
        return self._response

    async def __aexit__(self, *exc: object) -> bool:
        """Close the request without suppressing anything."""
        return False


class _FakeResponse:
    """Expose the status, JSON, and text of one canned response."""

    def __init__(self, payload: Any, *, is_json: bool = False, status: int = 200) -> None:
        """Set the canned payload and whether it is JSON."""
        self.status = status
        self._payload = payload
        self._is_json = is_json

    def raise_for_status(self) -> None:
        """Accept every canned response as successful."""

    async def json(self) -> Any:
        """Return the canned JSON payload."""
        return self._payload

    async def text(self, errors: str = "ignore") -> str:
        """Return the canned text payload."""
        return self._payload


class _FakeSession:
    """Answer repository, tree, directory, and raw-content requests from canned data."""

    def __init__(
        self,
        tree: list[str],
        bodies: dict[str, str],
        *,
        truncated: bool = False,
        directories: dict[str, list[dict]] | None = None,
    ) -> None:
        """Set the repository tree, the body per path, and the directory listings."""
        self.tree = tree
        self.bodies = bodies
        self.truncated = truncated
        self.directories = directories or {}
        self.contents_requests: list[str] = []

    def get(self, url: str) -> _FakeRequest:
        """Return the canned response for one request URL."""
        if url.endswith("/git/trees/main?recursive=1"):
            payload: Any = {
                "tree": [{"path": path, "type": "blob"} for path in self.tree],
                "truncated": self.truncated,
            }
            return _FakeRequest(_FakeResponse(payload, is_json=True))
        if url.endswith("/repos/owner/repo"):
            return _FakeRequest(_FakeResponse({"default_branch": "main"}, is_json=True))
        if "/contents/" in url:
            path = url.split("/contents/", 1)[1]
            self.contents_requests.append(path)
            if path in self.directories:
                return _FakeRequest(
                    _FakeResponse(self.directories[path], is_json=True)
                )
            return _FakeRequest(_FakeResponse({"message": "Not Found"}, status=404))
        return _FakeRequest(_FakeResponse(self.bodies[url.split("/main/", 1)[1]]))


class TestRepositoryEvidenceSelection(unittest.TestCase):
    """Keep source code in the evidence a bounded repository read produces."""

    def test_code_outranks_the_readme_and_every_other_document(self) -> None:
        """Read source, entry points, and tests before the README or any other prose."""
        self.assertLess(selection_priority("src/util.py"), selection_priority("README.md"))
        self.assertLess(selection_priority("src/main.py"), selection_priority("src/util.py"))
        self.assertLess(selection_priority("tests/test_app.py"), selection_priority("README.md"))
        self.assertLess(selection_priority("src/util.py"), selection_priority("docs/guide.md"))
        self.assertLess(selection_priority("src/util.py"), selection_priority("CHANGELOG.md"))
        self.assertLess(selection_priority("README.md"), selection_priority("docs/README.md"))
        self.assertLess(selection_priority("docs/guide.md"), selection_priority("docs/README.md"))
        self.assertTrue(is_documentation("docs/README.md"))
        self.assertFalse(is_documentation("src/util.py"))

    def test_a_read_still_includes_source_when_documentation_fills_the_tree(self) -> None:
        """Read code despite a repository whose documentation dwarfs its source files."""
        document = "Documentation line about the project.\n" * 200
        source = "print('value')\n" * 400
        tree = ["README.md", *(f"docs/README-{index}.md" for index in range(20)), "src/app.py", "src/util.py"]
        bodies = {
            "README.md": document,
            "src/app.py": source,
            "src/util.py": source,
            **{f"docs/README-{index}.md": document for index in range(20)},
        }

        dump = asyncio.run(
            collect_repository(
                EVIDENCE_URL, _FakeSession(tree, bodies), asyncio.Semaphore(8), {}
            )
        )

        self.assertIn("--- FILE: src/app.py ---", dump)
        self.assertIn("--- FILE: src/util.py ---", dump)
        files = _parse_metadata(dump).files
        documentation = sum(len(body) for name, _, body in files if is_documentation(name))
        self.assertLessEqual(documentation, MAX_DOC_CHARS)
        self.assertTrue(any(not is_documentation(name) for name, _, _ in files))

    def test_a_binary_body_is_recorded_but_never_reproduced(self) -> None:
        """Omit a weight blob the tree will not filter while keeping the real source."""
        tree = ["README.md", "src/app.py", "models/best_model.pt", "assets/weights"]
        bodies = {
            "README.md": "documentation line\n" * 6,
            "src/app.py": "print('a')\n" * 6,
            "assets/weights": BINARY_BODY,
        }

        dump = asyncio.run(
            collect_repository(
                EVIDENCE_URL, _FakeSession(tree, bodies), asyncio.Semaphore(8), {}
            )
        )

        self.assertIn("--- FILE: src/app.py ---", dump)
        self.assertIn("--- FILE: README.md ---", dump)
        self.assertNotIn("--- FILE: assets/weights ---", dump)
        self.assertNotIn("model_state_dict", dump)
        self.assertNotIn("best_model.pt", dump)

        metadata = _parse_metadata(dump)
        self.assertIn("assets/weights", metadata.not_included)
        self.assertTrue(metadata.truncated)

    def test_an_inline_payload_is_omitted_while_the_rest_of_its_file_is_kept(self) -> None:
        """Redact a stylesheet's embedded image but keep the rules and the coverage note."""
        tree = ["README.md", "style.css"]
        bodies = {
            "README.md": "documentation line\n" * 6,
            "style.css": (
                "body {\n  margin: 0;\n"
                "  background: url('data:image/png;base64," + "Zm9vYmFyQmF6" * 300 + "');\n}\n"
            ),
        }

        dump = asyncio.run(
            collect_repository(
                EVIDENCE_URL, _FakeSession(tree, bodies), asyncio.Semaphore(8), {}
            )
        )

        self.assertIn("--- FILE: style.css ---", dump)
        self.assertIn("margin: 0;", dump)
        self.assertNotIn("Zm9vYmFyQmF6", dump)
        self.assertIn("encoded payload omitted", dump)


class TestTruncatedTreeDiscovery(unittest.TestCase):
    """Read a repository's own files when GitHub truncates its recursive tree."""

    # A tree GitHub stopped early: the only candidate it listed is a stray root file, and
    # the rest of the listing sits under a vendored path the selector ignores.
    TRUNCATED_TREE: ClassVar[list[str]] = [".gitignore", "vendor/site-packages/pkg/module.py"]
    BODIES: ClassVar[dict[str, str]] = {
        ".gitignore": "*.log\n",
        "README.md": "# App\n",
        "package.json": '{"name": "app"}\n',
        "src/server.js": "const app = express();\n",
    }
    DIRECTORIES: ClassVar[dict[str, list[dict[str, str]]]] = {
        "": [
            {"path": "src", "type": "dir"},
            {"path": "node_modules", "type": "dir"},
            {"path": "package.json", "type": "file"},
        ],
        "src": [
            {"path": "src/server.js", "type": "file"},
            {"path": "src/assets", "type": "dir"},
        ],
    }

    def _read(self, session: _FakeSession) -> str:
        """Collect repository evidence through one fake session."""
        return asyncio.run(
            collect_repository(EVIDENCE_URL, session, asyncio.Semaphore(8), {})
        )

    def test_a_truncated_tree_is_listed_instead_of_scored_from_one_file(self) -> None:
        """Find the project's own files through the contents API when the tree is partial."""
        session = _FakeSession(
            self.TRUNCATED_TREE, self.BODIES, truncated=True, directories=self.DIRECTORIES
        )

        dump = self._read(session)

        self.assertIn("--- FILE: src/server.js ---", dump)
        self.assertIn("--- FILE: package.json ---", dump)
        self.assertIn("FILES FOUND: 3", dump)
        self.assertIn("FILES TRUNCATED: yes", dump)
        self.assertNotIn("module.py", dump)
        # The walk lists the root, then the project's own directories shallowest-first, and
        # never descends into an ignored vendored directory.
        self.assertEqual(session.contents_requests, ["", "src", "src/assets"])

    def test_a_complete_tree_costs_no_directory_requests(self) -> None:
        """Keep the walk out of the common case where the tree already describes the repo."""
        session = _FakeSession(
            ["README.md", "src/server.js"], self.BODIES, directories=self.DIRECTORIES
        )

        dump = self._read(session)

        self.assertIn("--- FILE: src/server.js ---", dump)
        self.assertNotIn("package.json", dump)
        self.assertEqual(session.contents_requests, [])

    def test_a_truncated_tree_without_a_listing_produces_no_report(self) -> None:
        """Refuse to score a repository the tree cannot describe and the API cannot list."""
        session = _FakeSession(self.TRUNCATED_TREE, self.BODIES, truncated=True)

        dump = self._read(session)

        self.assertTrue(dump.startswith("Error:"), dump)
        self.assertIn("truncated the file tree", dump)
        self.assertIn("HTTP 404", dump)
        self.assertNotIn("--- FILE:", dump)

    def test_the_walk_stops_at_its_depth_limit(self) -> None:
        """Leave paths deeper than the depth limit to the tree listing."""
        directories = {
            "": [{"path": "a", "type": "dir"}],
            "a": [{"path": "a/b", "type": "dir"}],
            "a/b": [{"path": "a/b/c", "type": "dir"}],
            "a/b/c": [{"path": "a/b/c/d", "type": "dir"}],
            "a/b/c/d": [{"path": "a/b/c/d/e", "type": "dir"}],
            "a/b/c/d/e": [{"path": "a/b/c/d/e/deep.py", "type": "file"}],
        }
        session = _FakeSession(
            self.TRUNCATED_TREE, self.BODIES, truncated=True, directories=directories
        )

        dump = self._read(session)

        self.assertIn(".gitignore", dump)
        self.assertNotIn("deep.py", dump)
        self.assertNotIn("a/b/c/d/e", session.contents_requests)

    def test_the_walk_respects_its_request_budget(self) -> None:
        """Stop listing once the request budget is spent against GitHub's quota."""
        session = _FakeSession([], self.BODIES, directories=self.DIRECTORIES)
        fetch = make_fetcher(session, asyncio.Semaphore(8), {})

        found, listed_root = asyncio.run(
            walk_directories(fetch, "owner", "repo", max_requests=1)
        )

        self.assertTrue(listed_root)
        self.assertEqual(found, ["package.json"])
        self.assertEqual(session.contents_requests, [""])


class TestRepositoryReaderFiltering(unittest.TestCase):
    """Check the file filter independently of GitHub or an active Ollama server."""

    def test_assets_and_lockfiles_are_excluded_but_source_and_docs_remain(self) -> None:
        """Keep useful source and documentation while dropping bulky assets and locks."""
        self.assertFalse(is_candidate("vendor/image.png"))
        self.assertFalse(is_candidate("package-lock.json"))
        self.assertTrue(is_candidate("src/main.py"))
        self.assertTrue(is_candidate("README.md"))

    def test_model_weights_and_media_are_excluded_before_they_are_fetched(self) -> None:
        """Reject trained weights, media, and serialized data by their extensions."""
        self.assertFalse(is_candidate("models/best_model.pt"))
        self.assertFalse(is_candidate("models/hand_landmarker.task"))
        self.assertFalse(is_candidate("assets/theme.mp3"))
        self.assertFalse(is_candidate("data/cache.pkl"))
        self.assertFalse(is_candidate("src/example.min.js"))
        self.assertTrue(is_candidate("src/gamemain.py"))
        self.assertTrue(is_candidate("assets/weights"))

    def test_a_binary_body_is_detected_without_relying_on_its_extension(self) -> None:
        """Judge the fetched bytes so an extensionless blob cannot slip through."""
        self.assertTrue(looks_binary(BINARY_BODY))
        self.assertTrue(looks_binary("\x10\x11\x12" * 20))
        self.assertFalse(looks_binary(""))
        self.assertFalse(looks_binary("print('ok')\n# a comment\n"))
        self.assertFalse(looks_binary("café — an emoji \U0001f389 and a tab\tindent\n"))

    def test_an_encoded_payload_inside_readable_text_is_replaced_by_a_marker(self) -> None:
        """Strip an inline data URI so its bytes cannot be quoted as authorship evidence."""
        payload = "data:image/jpeg;base64," + "A1b2C3d4E5f6" * 200
        stylesheet = f"h1 {{\n  background-image: url('{payload}');\n}}\n"

        redacted = redact_payloads(stylesheet)

        self.assertIn("background-image", redacted)
        self.assertIn("data:image/jpeg;base64,", redacted)
        self.assertNotIn("A1b2C3d4E5f6", redacted)
        self.assertIn("encoded payload omitted: 2400 character(s)", redacted)
        self.assertLess(len(redacted), len(stylesheet))

    def test_ordinary_source_and_short_tokens_survive_redaction(self) -> None:
        """Leave readable code alone, including tokens shorter than a payload."""
        source = (
            "def add(a, b):  # keeps spacing, quotes and `backticks`\n"
            "    token = '{\"dGhpcyBpcyBnZW5lcmF0ZWQgZGF0YQ==\"}'\n"
            "    return a + b\n"
        )

        self.assertEqual(redact_payloads(source), source)


class TestCitationGrounding(unittest.TestCase):
    """Reject evidence a served view cannot support."""

    VIEW = (
        "REPOSITORY: owner/repo\nDEFAULT BRANCH: main\nFILES FOUND: 2\nFILES FETCHED: 2\n"
        "FILES INCLUDED (2): app.py, util.py\nFILES TRUNCATED: no\n\n"
        "--- FILE: app.py ---\ndef add(a, b):\n    return a+b\n\n"
        "--- FILE: util.py ---\nprint(add(1, 2))\n"
    )

    # A view that shows the project's description and its dependency manifest beside one
    # code file: the first two are what every project of that shape carries.
    PACKAGING_VIEW = (
        "REPOSITORY: owner/repo\nFILES FOUND: 3\nFILES FETCHED: 3\n"
        "FILES INCLUDED (3): README.md, package.json, app.py\nFILES TRUNCATED: no\n\n"
        "--- FILE: README.md ---\n# Project\n\nA tiny module.\n\n"
        '--- FILE: package.json ---\n{"name": "app", "version": "1.0.0"}\n\n'
        "--- FILE: app.py ---\ndef add(a, b):\n    return a+b\n"
    )

    @staticmethod
    def _report(evidence: str) -> str:
        """Render a three-part report whose Evidence line carries the given entry."""
        return (
            "1. AI Involvement Score (0-100%): 60%\n"
            f"Ranked AI evidence (strongest to weakest):\nRank 1: {evidence}\n\n"
            "2. Code Quality Score (0-100%): 55%\n"
            "- Deductions: -15% no test files\n\n"
            "3. Summary & Notable Findings: A short module with two files.\n"
        )

    def test_a_quote_from_a_reproduced_file_is_supported(self) -> None:
        """Accept evidence that quotes a shown file and names it."""
        self.assertEqual(
            verify_citations(self._report("`def add(a, b):` in app.py"), self.VIEW), []
        )

    def test_a_path_inside_quoted_content_is_not_read_as_a_cited_file(self) -> None:
        """Accept a quote whose text carries a path the view never reproduced.

        Quoted source routinely contains a path in an attribute or a route, and treating
        that slash as a cited file name rejected evidence the view really had shown.
        """
        view = (
            "REPOSITORY: owner/repo\nFILES FOUND: 1\nFILES FETCHED: 1\n"
            "FILES INCLUDED (1): page.html\nFILES TRUNCATED: no\n\n"
            "--- FILE: page.html ---\n"
            '<audio id="bg" src="songs/theme.mp3"></audio>\n'
        )
        unsupported = verify_citations(
            self._report(
                "the page plays a local file, for example "
                '`<audio id="bg" src="songs/theme.mp3">` in page.html'
            ),
            view,
        )

        self.assertEqual(unsupported, [])

    def test_a_comment_marker_quote_is_not_read_as_a_cited_file(self) -> None:
        """Accept a quoted comment whose leading ``//`` would otherwise read as a path.

        A quote of a Go or C++ line comment carries a slash but names no file, so it must
        not be checked against the view's file list as if it were one.
        """
        view = (
            "REPOSITORY: owner/repo\nFILES FOUND: 1\nFILES FETCHED: 1\n"
            "FILES INCLUDED (1): pkg/asq/nodes_test.go\nFILES TRUNCATED: no\n\n"
            "--- FILE: pkg/asq/nodes_test.go ---\n//asq_start\nfunc TestNodes() {}\n"
        )
        unsupported = verify_citations(
            self._report("`//asq_start` in pkg/asq/nodes_test.go"), view
        )

        self.assertEqual(unsupported, [])

    def test_a_cited_file_with_a_long_extension_is_read_in_full(self) -> None:
        """Read ``in Main.storyboard`` as the whole name, not as a six-letter prefix of it.

        A cap on the extension length read the name as ``Main.storyb``, and the citation check
        then reported a file the view did reproduce as missing.
        """
        from contracts.parsing import cited_files

        self.assertEqual(
            cited_files("the layout sets a colour in ios/Runner/Base.lproj/Main.storyboard"),
            ["ios/Runner/Base.lproj/Main.storyboard"],
        )

    def test_prose_inside_a_quote_is_not_read_as_a_cited_file(self) -> None:
        """Accept a code quote whose own text says "in <dotted name>".

        A quote is repository content, and code says "in" constantly: `for line in
        os.environ.get(...)` was read as a citation of a file named os.environ.get, which
        rejected a real quote from a file the view had reproduced. Only the prose around a
        quote can name a file.
        """
        view = (
            "REPOSITORY: owner/repo\nFILES FOUND: 1\nFILES FETCHED: 1\n"
            "FILES INCLUDED (1): scripts/run.py\nFILES TRUNCATED: no\n\n"
            "--- FILE: scripts/run.py ---\n"
            'for line in os.environ.get("R2_ENV", "").splitlines():\n'
        )
        unsupported = verify_citations(
            self._report(
                "one source line is a comment, for example `for line in "
                'os.environ.get("R2_ENV", "").splitlines():` in scripts/run.py'
            ),
            view,
        )

        self.assertEqual(unsupported, [])

    def test_a_bare_backticked_path_the_view_lacks_is_still_rejected(self) -> None:
        """Keep failing a citation whose quoted token is a path the view never showed."""
        view = self.VIEW + "\n--- FILE: notes.txt ---\nsee lib/helper.py for the wrapper\n"
        unsupported = verify_citations(
            self._report(
                "the project vendored a helper, for example `lib/helper.py` in app.py"
            ),
            view,
        )

        self.assertTrue(unsupported)
        self.assertIn("names a file this view does not reproduce", unsupported[0])

    def test_the_payload_omission_marker_is_not_citable_evidence(self) -> None:
        """Refuse an entry that quotes a marker describing the view instead of the code."""
        view = self.VIEW + "\n[... encoded payload omitted: 2400 character(s) ...]\n"
        unsupported = verify_citations(
            self._report(
                "the stylesheet is mostly machine output, for example `[... encoded "
                "payload omitted: 2400 character(s) ...]` in app.py"
            ),
            view,
        )

        self.assertTrue(unsupported)
        self.assertIn("bookkeeping", unsupported[0])

    def test_a_fabricated_filename_cannot_ride_on_a_real_quote(self) -> None:
        """Fail an entry that quotes real text but names a file the view never showed."""
        unsupported = verify_citations(
            self._report("`def add(a, b):` in snippet.py"), self.VIEW
        )

        self.assertEqual(len(unsupported), 1)
        self.assertIn("snippet.py", unsupported[0])

    def test_a_quote_from_another_file_does_not_support_the_named_file(self) -> None:
        """Fail a real quote attributed to a file that does not contain it.

        The quote exists in util.py, so the view-wide check passes; the entry names app.py,
        which never shows that line, and it is not evidence about app.py.
        """
        unsupported = verify_citations(
            self._report("`print(add(1, 2))` in app.py"), self.VIEW
        )

        self.assertEqual(len(unsupported), 1)
        self.assertIn("absent from the file it names", unsupported[0])

    def test_a_quote_inside_the_named_file_is_supported(self) -> None:
        """Accept the same quote when it is attributed to the file that shows it."""
        self.assertEqual(
            verify_citations(self._report("`print(add(1, 2))` in util.py"), self.VIEW),
            [],
        )

    def test_the_project_description_is_not_authorship_evidence(self) -> None:
        """Refuse an entry whose only named file is the project's own description."""
        unsupported = verify_citations(
            self._report(
                "the README is fully structured, for example `# Project` in README.md"
            ),
            self.PACKAGING_VIEW,
        )

        self.assertEqual(len(unsupported), 1)
        self.assertIn("description or tooling", unsupported[0])

    def test_a_dependency_manifest_is_not_authorship_evidence(self) -> None:
        """Refuse an entry whose only named file is a manifest every project carries."""
        unsupported = verify_citations(
            self._report(
                "the manifest looks generated, for example "
                '`{"name": "app", "version": "1.0.0"}` in package.json'
            ),
            self.PACKAGING_VIEW,
        )

        self.assertEqual(len(unsupported), 1)
        self.assertIn("description or tooling", unsupported[0])

    def test_an_entry_grounded_in_code_beside_packaging_stays_supported(self) -> None:
        """Keep an entry that names a code file even when the view also shows packaging."""
        self.assertEqual(
            verify_citations(
                self._report("`def add(a, b):` in app.py is the whole module"),
                self.PACKAGING_VIEW,
            ),
            [],
        )

    def test_published_evidence_keeps_only_code_grounded_entries(self) -> None:
        """Drop a packaging-only entry from the evidence a report publishes."""
        report = self._report(
            "the README is fully structured, for example `# Project` in README.md"
        ).replace(
            "\n\n2. Code Quality",
            "\nRank 2: `def add(a, b):` in app.py is a single untyped definition\n\n"
            "2. Code Quality",
        )

        self.assertEqual(
            authorship_evidence(report),
            ["`def add(a, b):` in app.py is a single untyped definition"],
        )

    def test_a_style_count_is_not_authorship_evidence(self) -> None:
        """Refuse an entry whose reason rests on formatting the view does not publish.

        The stored gallery run's strongest sign was "0 line(s) are tab indented, for example
        `using UnityEngine;` in CatClicked.cs": the quote was real, so the quotation check
        alone accepted it, and the number it rested on came from a measurement block that no
        longer carries one.
        """
        unsupported = verify_citations(
            self._report(
                "0 line(s) are tab indented, for example `def add(a, b):` in app.py"
            ),
            self.VIEW,
        )

        self.assertEqual(len(unsupported), 1)
        self.assertIn("formatting", unsupported[0])

    def test_a_style_count_is_dropped_from_the_published_evidence(self) -> None:
        """Keep a formatting-only entry out of the evidence a report publishes."""
        report = self._report(
            "0 line(s) are tab indented, for example `def add(a, b):` in app.py"
        ).replace(
            "\n\n2. Code Quality",
            "\nRank 2: `print(add(1, 2))` in util.py does not validate its input\n\n"
            "2. Code Quality",
        )

        self.assertEqual(
            authorship_evidence(report),
            ["`print(add(1, 2))` in util.py does not validate its input"],
        )

    def test_bookkeeping_is_not_evidence(self) -> None:
        """Refuse a coverage header as the quotation supporting a claim."""
        unsupported = verify_citations(
            self._report("`FILES FOUND: 2` shows a generated layout"), self.VIEW
        )

        self.assertEqual(len(unsupported), 1)
        self.assertIn("bookkeeping", unsupported[0])

    def test_a_grounded_banner_quote_still_counts(self) -> None:
        """Keep a comment quoting an authorship banner valid: a comment is real content."""
        view = self.VIEW.replace(
            "def add(a, b):", "# generated by ChatGPT\ndef add(a, b):"
        )

        self.assertEqual(
            verify_citations(self._report("`# generated by ChatGPT` in app.py"), view), []
        )

    def test_an_impression_is_not_evidence_even_beside_a_real_quote(self) -> None:
        """Refuse 'reads as machine-written' even when the entry also quotes a real line."""
        unsupported = verify_citations(
            self._report(
                "the code reads as machine-written, for example `def add(a, b):` in app.py"
            ),
            self.VIEW,
        )

        self.assertEqual(len(unsupported), 1)
        self.assertIn("impression", unsupported[0])

    def test_an_absence_claim_is_not_evidence_even_beside_a_real_quote(self) -> None:
        """Refuse '0 human traces' as evidence when the entry also quotes a real line."""
        unsupported = verify_citations(
            self._report(
                "0 human trace(s) appear in the source, for example "
                "`def add(a, b):` in app.py"
            ),
            self.VIEW,
        )

        self.assertEqual(len(unsupported), 1)
        self.assertIn("impression", unsupported[0])

    def test_a_counted_trace_is_a_measurement_not_an_absence(self) -> None:
        """Accept a counted trace: '3 human trace(s)' reports what the source contains."""
        self.assertEqual(
            verify_citations(
                self._report(
                    "3 human trace(s) appear in the source, for example "
                    "`def add(a, b):` in app.py"
                ),
                self.VIEW,
            ),
            [],
        )

    def test_ranked_evidence_is_parsed_in_the_order_the_judge_reports(self) -> None:
        """Parse numbered judge evidence as ordered claims without treating ranks as citations."""
        report = (
            "1. AI Involvement Score (0-100%): 62%\n"
            "Ranked AI evidence (strongest to weakest):\n"
            "Rank 1: `# generated by ChatGPT` in app.py, explicit author attribution.\n"
            "Rank 2: `async def main` in app.py, weak style clue.\n\n"
            "2. Code Quality Score (0-100%): 80%\n"
            "- Deductions: no deductions applied\n\n"
            "3. Summary & Notable Findings: A compact application with a clear entry point.\n"
        )

        self.assertEqual(
            evidence_claims(report),
            [
                "`# generated by ChatGPT` in app.py, explicit author attribution.",
                "`async def main` in app.py, weak style clue.",
            ],
        )
        self.assertEqual(parse_report(report)["ai_score"], 62)

    def test_ranked_evidence_must_have_consecutive_ranks_from_one(self) -> None:
        """Reject missing, repeated, or out-of-order rank labels in an otherwise valid report."""
        template = (
            "1. AI Involvement Score (0-100%): 62%\n"
            "Ranked AI evidence (strongest to weakest):\n"
            "{entries}\n\n"
            "2. Code Quality Score (0-100%): 80%\n"
            "- Deductions: no deductions applied\n\n"
            "3. Summary & Notable Findings: A compact application with a clear entry point.\n"
        )
        for entries in (
            "Rank 2: `def add(a, b):` in app.py",
            "Rank 1: `def add(a, b):` in app.py\nRank 1: `return a+b` in app.py",
            "Rank 1: `def add(a, b):` in app.py\nRank 3: `return a+b` in app.py",
        ):
            with self.subTest(entries=entries):
                self.assertFalse(parse_report(template.format(entries=entries))["valid"])

    def test_empty_ranked_evidence_requires_an_exact_no_evidence_sentinel(self) -> None:
        """Reject unranked prose but accept the explicit no-evidence report."""
        prose = (
            "1. AI Involvement Score (0-100%): 10%\n"
            "Ranked AI evidence (strongest to weakest):\n"
            "No relevant signs were noticed in this code.\n\n"
            "2. Code Quality Score (0-100%): 80%\n"
            "- Deductions: no deductions applied\n\n"
            "3. Summary & Notable Findings: A compact application with a clear entry point.\n"
        )

        self.assertFalse(parse_report(prose)["valid"])
        self.assertTrue(parse_report(self._report("No AI-specific evidence found."))["valid"])

    def test_an_impression_word_inside_a_quote_is_content(self) -> None:
        """Keep a quoted line that says 'reads as': only the claim wording is judged."""
        view = self.VIEW.replace(
            "def add(a, b):", "# reads as machine-written\ndef add(a, b):"
        )

        self.assertEqual(
            verify_citations(self._report("`# reads as machine-written` in app.py"), view),
            [],
        )

    def test_a_quote_the_view_does_not_contain_is_unsupported(self) -> None:
        """Fail the invented banner quoted on a repository that carries none."""
        unsupported = verify_citations(
            self._report("`# This code was generated by ChatGPT` in app.py"), self.VIEW
        )

        self.assertEqual(len(unsupported), 1)
        self.assertIn("absent from the view", unsupported[0])

    def test_an_entry_that_quotes_nothing_is_not_silently_accepted(self) -> None:
        """Require a quotation: a bare verdict is not checkable evidence."""
        unsupported = verify_citations(
            self._report("the code reads as machine-written"), self.VIEW
        )

        self.assertEqual(len(unsupported), 1)
        self.assertIn("quotes no text", unsupported[0])

    def test_no_evidence_sentinel_is_not_a_citation(self) -> None:
        """Leave the explicit no-evidence sentinel out of citation validation."""
        report = self._report("no AI-specific evidence found")

        self.assertNotIn("no AI-specific evidence found", extract_evidence(report))
        self.assertEqual(verify_citations(report, self.VIEW), [])

    def test_deductions_are_not_required_to_quote_code(self) -> None:
        """Keep score adjustments exempt: they price the work, not the repository's text."""
        report = (
            "1. AI Involvement Score (0-100%): 40%\n"
            "Ranked AI evidence (strongest to weakest):\n"
            "Rank 1: `def add(a, b):` in app.py\n\n"
            "2. Code Quality Score (0-100%): 60%\n"
            "- Deductions: -15% no README with setup instructions\n\n"
            "3. Summary & Notable Findings: A short module with two files.\n"
        )

        self.assertEqual(verify_citations(report, self.VIEW), [])


class TestJudgeJustificationContract(unittest.TestCase):
    """Require a stated reason for each score without rejecting a report over it."""

    REPORT = (
        "1. AI Involvement Score (0-100%): 62%\n"
        "- Why this score: the files shown carry no generator marker and one author-specific "
        "detail, so the number rests on `def add(a, b):` in app.py.\n"
        "Ranked AI evidence (strongest to weakest):\n"
        "Rank 1: `def add(a, b):` in app.py is a single untyped definition with no docstring.\n\n"
        "2. Code Quality Score (0-100%): 80%\n"
        "- Why this score: `return a+b` in app.py handles no error, which caps the number "
        "until the function validates its inputs.\n"
        "- Deductions: no deductions applied\n\n"
        "3. Summary & Notable Findings: A two-file module whose one definition carries no "
        "docstring.\n"
    )

    def test_the_instructions_ask_for_a_reason_for_each_score(self) -> None:
        """Ask the judge to justify both scores and to weigh code over the README."""
        self.assertEqual(SYSTEM_PROMPT.count("- Why this score:"), 2)
        self.assertIn("cannot raise either score", SYSTEM_PROMPT)
        self.assertIn("- Why this score:", REPORT_INSTRUCTION)
        self.assertIn("never rank a README", REPORT_INSTRUCTION)

    def test_the_instructions_rule_out_packaging_and_formatting_evidence(self) -> None:
        """Name the files that cannot carry a sign and refuse indentation as one."""
        self.assertIn("must not be ranked at all", SYSTEM_PROMPT)
        for name in ("package.json", "pyproject.toml", "requirements.txt", ".gitignore"):
            with self.subTest(name=name):
                self.assertIn(name, SYSTEM_PROMPT)
        self.assertIn("Formatting is not evidence either", SYSTEM_PROMPT)
        self.assertIn("never cite formatting", REPORT_INSTRUCTION)

    def test_a_report_carrying_its_justifications_parses(self) -> None:
        """Read both scores from a report that states a reason for each of them."""
        parsed = parse_report(self.REPORT)

        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["ai_score"], 62)
        self.assertEqual(parsed["code_score"], 80)

    def test_a_justification_inside_the_evidence_block_is_not_an_entry(self) -> None:
        """Skip a justification the judge placed after the ranks, keeping the report valid."""
        report = self.REPORT.replace(
            "\n\n2. Code Quality",
            "\n- Why this score: the view shows no test file beside the module.\n\n"
            "2. Code Quality",
        )

        self.assertTrue(parse_report(report)["valid"])
        self.assertEqual(len(evidence_claims(report)), 1)


class TestPartialBatchAggregation(unittest.TestCase):
    """Keep one unusable evidence batch from discarding a repository's evaluation."""

    @staticmethod
    def _report(content: str, lines: int = 10) -> dict:
        """Describe one batch result the way the evaluator hands it to aggregation."""
        return {
            "content": content,
            "parsed": parse_report(content),
            "unsupported_citations": [],
            "lines": lines,
        }

    def test_a_failed_batch_is_recorded_without_losing_the_other_batches(self) -> None:
        """Score the batches that validated and name the batch that did not."""
        result = combine_reports(
            [
                self._report(VALID_REPORT_A, lines=100),
                self._report(INVALID_REPORT, lines=50),
                self._report(VALID_REPORT_B, lines=200),
            ],
            [100, 50, 200],
        )

        self.assertEqual(result["status"], 200)
        self.assertEqual(parse_report(result["content"])["ai_score"], 77)
        self.assertEqual(len(result["failed_batches"]), 1)
        self.assertEqual(result["failed_batches"][0]["index"], 1)
        self.assertEqual(result["failed_batches"][0]["content"], INVALID_REPORT)
        self.assertEqual(result["batch_count"], 3)
        self.assertEqual(len(result["batch_reports"]), 3)

    def test_a_failed_batch_cannot_discard_the_score_or_add_citations(self) -> None:
        """Keep a discarded batch's claims out of the aggregate it was dropped from."""
        failed = self._report(INVALID_REPORT, lines=500)
        failed["unsupported_citations"] = ["`# generated by ChatGPT` in app.py"]

        result = combine_reports(
            [self._report(VALID_REPORT_A, lines=100), failed],
            [100, 500],
        )

        self.assertEqual(result["status"], 200)
        self.assertEqual(result["unsupported_citations"], [])
        self.assertEqual(parse_report(result["content"])["ai_score"], 70)

    def test_the_aggregate_leads_with_the_strongest_batch_in_code_only(self) -> None:
        """Rank published evidence by the batch whose score it carried, quoting code only."""
        weaker = self._report(VALID_REPORT_A, lines=400)
        stronger = self._report(
            VALID_REPORT_B.replace(
                "Rank 1: `print('b')` in util.py repeats one statement twice",
                "Rank 1: the README lists the modules, for example `# Project` in README.md\n"
                "Rank 2: `print('b')` in util.py repeats one statement twice",
            ),
            lines=100,
        )

        result = combine_reports([weaker, stronger], [400, 100])

        entries = evidence_claims(result["content"])
        self.assertEqual(len(entries), 2)
        self.assertIn("util.py", entries[0])
        self.assertIn("app.py", entries[1])

    def test_a_repository_whose_every_batch_failed_still_fails(self) -> None:
        """Report a repository-level failure when no batch produced a report."""
        result = combine_reports(
            [self._report(INVALID_REPORT), self._report(INVALID_REPORT)],
            [1, 1],
        )

        self.assertEqual(result["status"], 502)
        self.assertIn("None of the 2", result["content"])
        self.assertEqual(len(result["failed_batches"]), 2)


class TestPartialEvidenceRuntime(unittest.TestCase):
    """Grade a repository whose evidence batches do not all come back usable."""

    def _grader(self) -> grader_runtime.Model:
        """Create an Ollama grader that reports only the released model tag."""
        with patch("grading.ollama.Client") as client_factory:
            client_factory.return_value = _ListClient(["hackgrader:latest"])
            grader = grader_runtime.Model(readiness_timeout=0.1)
        grader._list_client = _ListClient(["hackgrader:latest"])
        return grader

    def test_one_unusable_batch_is_recorded_without_failing_the_repository(self) -> None:
        """Score usable batches and retain the failed batch on the result."""
        grader = self._grader()
        replies = [
            SimpleNamespace(message=SimpleNamespace(content=VALID_REPORT_A)),
            SimpleNamespace(message=SimpleNamespace(content=VALID_REPORT_B)),
            SimpleNamespace(message=SimpleNamespace(content=INVALID_REPORT)),
        ]
        with (
            patch.object(grader, "wait_for_ollama", return_value=True),
            patch.object(grader, "_available_model", return_value=True),
            patch("grading.runtime.read_repository", return_value=PARTIAL_VIEWS[0]),
            patch("grading.runtime.split_dump", return_value=list(PARTIAL_VIEWS)),
            patch.object(grader, "count_prompt_tokens", return_value=1),
            patch.object(grader, "_chat", side_effect=replies),
        ):
            result = grader.query(REPO_QUESTION)

        # A view with no source lines still carries weight 1, exactly as aggregation weights it.
        weights = [max(source_line_count(view), 1) for view in PARTIAL_VIEWS]
        expected = round(
            (70 * weights[0] + 80 * weights[2]) / (weights[0] + weights[2])
        )
        self.assertEqual(result["status"], 200)
        self.assertEqual(set(result["analysis"]), {"metrics", "coverage"})
        self.assertNotIn("ai_score", result["analysis"])
        self.assertNotIn("quality_score", result["analysis"])
        self.assertEqual(len(result["failed_batches"]), 1)
        self.assertEqual(result["failed_batches"][0]["index"], 2)
        self.assertEqual(result["failed_batches"][0]["content"], INVALID_REPORT)
        self.assertEqual(len(result["batch_reports"]), 3)
        self.assertEqual(parse_report(result["content"])["ai_score"], expected)

        annotated = Reporting()._annotate(result)
        self.assertEqual(annotated["status"], 200)
        self.assertEqual(annotated["failed_batches"], result["failed_batches"])
        self.assertNotIn("flags", annotated)
        self.assertNotIn("report_findings", annotated)
        self.assertEqual(set(annotated["analysis"]), {"metrics", "coverage"})
        self.assertEqual(set(annotated["analysis"]), {"metrics", "coverage"})


class TestNoAggregationSubstitution(unittest.TestCase):
    """Aggregate the model's own scores without replacing them on a citation failure."""

    def test_unsupported_citations_do_not_replace_the_aggregate_score(self) -> None:
        """Publish the line-weighted model score and name the unsupported entries."""
        cited = {
            "content": VALID_REPORT_A,
            "parsed": parse_report(VALID_REPORT_A),
            "unsupported_citations": ["`# generated by ChatGPT` in app.py"],
            "lines": 100,
        }

        result = combine_reports([cited], [100])

        self.assertEqual(result["status"], 200)
        self.assertEqual(parse_report(result["content"])["ai_score"], 70)
        self.assertEqual(
            result["unsupported_citations"], ["`# generated by ChatGPT` in app.py"]
        )


class TestJudgeOwnedScores(unittest.TestCase):
    """Publish judge scores without using detector measurements or automated flags."""

    REPORT = (
        "1. AI Involvement Score (0-100%): 74%\n"
        "Ranked AI evidence (strongest to weakest):\n"
        "Rank 1: `# generated by ChatGPT` in app.py, explicit AI attribution.\n"
        "Rank 2: `async def main` in app.py, weak style clue only.\n\n"
        "2. Code Quality Score (0-100%): 68%\n"
        "- Deductions: -12% no README with setup instructions; -6% no tests\n\n"
        "3. Summary & Notable Findings: A stylesheet and a quiz page with eleven screens.\n"
    )

    def _project(
        self, detector_ai: int, detector_code: int = 85, findings: list[dict] | None = None
    ) -> dict:
        """Build one stored grading result with a judge report and a measurement."""
        return {
            "status": 200,
            "content": self.REPORT,
            "analysis": {
                "metrics": {"chars": 8200},
                "coverage": {"files_shown": 4, "files_fetched": 4, "files_found": 4},
            },
        }

    def test_detector_disagreement_does_not_change_scores_or_create_flags(self) -> None:
        """Publish judge numbers even when prompt measurements disagree, with no flags."""
        annotated = Reporting()._annotate(self._project(detector_ai=2))

        self.assertEqual(annotated["ai_score"], 74)
        self.assertEqual(parse_report(annotated["content"])["ai_score"], 74)
        self.assertNotIn("ai_score", annotated["analysis"])
        self.assertNotIn("ai_findings", annotated["analysis"])
        self.assertNotIn("detector_ai_score", annotated)
        self.assertNotIn("flags", annotated)
        self.assertNotIn("report_findings", annotated)
        self.assertEqual(set(annotated["analysis"]), {"metrics", "coverage"})

    def test_detector_quality_measurement_does_not_replace_judge_score(self) -> None:
        """Report judge quality score regardless of prompt measurement."""
        annotated = Reporting()._annotate(self._project(detector_ai=45, detector_code=95))

        self.assertEqual(annotated["code_score"], 68)
        self.assertEqual(parse_report(annotated["content"])["code_score"], 68)
        self.assertNotIn("quality_score", annotated["analysis"])
        self.assertNotIn("detector_code_score", annotated)
        self.assertNotIn("flags", annotated)

    def test_the_reported_evidence_is_judge_ranked_and_keeps_its_order(self) -> None:
        """Keep the judge's ranked evidence order and supporting explanations."""
        annotated = Reporting()._annotate(self._project(detector_ai=45))

        self.assertEqual(
            annotated["evidence"],
            [
                "`# generated by ChatGPT` in app.py, explicit AI attribution.",
                "`async def main` in app.py, weak style clue only.",
            ],
        )
        self.assertEqual(
            annotated["deductions"],
            ["-12% no README with setup instructions", "-6% no tests"],
        )
        self.assertEqual(annotated["analysis"]["metrics"], {"chars": 8200})
        self.assertEqual(annotated["analysis"]["coverage"]["files_shown"], 4)
        self.assertNotIn("ai_score", annotated["analysis"])
        self.assertNotIn("ai_findings", annotated["analysis"])

    def test_detector_signals_are_not_retained_as_results(self) -> None:
        """Keep the judge score without retaining the detector verdict or findings."""
        annotated = Reporting()._annotate(
            self._project(
                detector_ai=88, findings=[{"weight": 60, "reason": "explicit marker"}]
            )
        )

        self.assertEqual(annotated["ai_score"], 74)
        self.assertNotIn("ai_score", annotated["analysis"])
        self.assertNotIn("ai_findings", annotated["analysis"])

    def test_unsupported_citations_do_not_lower_the_reported_score(self) -> None:
        """Keep the judge's number when the view does not support its citations."""
        project = self._project(detector_ai=2)
        project["unsupported_citations"] = ["`# generated by ChatGPT` in app.py"]

        annotated = Reporting()._annotate(project)

        self.assertEqual(annotated["ai_score"], 74)
        self.assertNotIn("flags", annotated)
        self.assertNotIn("ai_score", annotated["analysis"])

    def test_legacy_analysis_verdicts_are_removed_but_context_survives(self) -> None:
        """Strip stored detector judgments while retaining its factual prompt context."""
        annotated = Reporting()._annotate(
            {
                "status": 200,
                "content": self.REPORT,
                "analysis": {
                    "ai_score": 99,
                    "quality_score": 1,
                    "ai_findings": [{"reason": "heuristic"}],
                    "metrics": {"chars": 300},
                    "coverage": {"files_shown": 2},
                },
            }
        )

        self.assertEqual(annotated["ai_score"], 74)
        self.assertEqual(annotated["code_score"], 68)
        self.assertEqual(annotated["analysis"], {
            "metrics": {"chars": 300},
            "coverage": {"files_shown": 2},
        })

    def test_an_unparseable_report_publishes_no_score(self) -> None:
        """Report no score at all rather than a measurement made by something else."""
        annotated = Reporting()._annotate(
            {
                "status": 502,
                "content": "no report here",
                "analysis": {"ai_score": 2, "quality_score": 90, "metrics": {"chars": 50}},
            }
        )

        self.assertIsNone(annotated["ai_score"])
        self.assertIsNone(annotated["code_score"])
        self.assertNotIn("detector_ai_score", annotated)
        self.assertNotIn("detector_code_score", annotated)
        self.assertEqual(annotated["status"], 502)
        self.assertNotIn("ai_score", annotated["analysis"])
        self.assertEqual(annotated["analysis"]["metrics"], {"chars": 50})

    def test_a_judge_score_without_a_detector_analysis_is_published(self) -> None:
        """Publish the judge's numbers when no measurement accompanied the report."""
        annotated = Reporting()._annotate({"status": 200, "content": self.REPORT})

        self.assertEqual(annotated["ai_score"], 74)
        self.assertEqual(annotated["code_score"], 68)
        self.assertNotIn("ai_score", annotated["analysis"])
        self.assertNotIn("detector_ai_score", annotated)
        self.assertNotIn("flags", annotated)


if __name__ == "__main__":
    unittest.main()
