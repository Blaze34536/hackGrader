"""Generate every candidate row: AIGCodeSet, real repositories, derived, and artifacts.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from typing import Any

import contracts as report
from data.validation import DEFAULT_MAX_LEN, canonical_source_url, validate_row

from ..config import (
    ARTIFACT_ROWS,
    MIN_REAL_REPOSITORIES_PER_SPLIT,
    SNIPPET_ONLY,
    SYNTHETIC_REPOS,
)
from ..examples import aig_samples, snippet_rows
from ..features import feature_weights
from ..labels import (
    LabelManifest,
    load_repository_labels,
    report_manifest,
    require_usable,
)
from ..publishing import candidate_tokenizer
from ..rendering import finalize, usable
from ..sources import repo_fetcher, repo_rows
from ..synthetic import artifact_rows, synthetic_repo_rows


def repository_manifest(path: str | None) -> LabelManifest | None:
    """Load verified repository labels, or return ``None`` when none were requested."""
    if not path:
        return None
    return load_repository_labels(path)


def real_repositories(
    path: str | None, labels: LabelManifest | None = None
) -> list[dict[str, Any]]:
    """Load repository rows when a real-repository source list was supplied."""
    if SNIPPET_ONLY and path:
        raise SystemExit(
            "[!] HACKGRADER_SNIPPET_ONLY=1 excludes real-repository rows, so --repos "
            "contradicts it; drop one of the two."
        )
    if not path:
        if SNIPPET_ONLY:
            print(
                "[i] snippet-only build: no real-repository rows, by explicit request."
            )
            return []
        raise SystemExit(
            "[!] --repos is required: real-repository examples are required for all "
            "splits. To build without them, set HACKGRADER_SNIPPET_ONLY=1 and expect a "
            "model without the repository head."
        )
    if labels is None:
        # The style reading is not authorship evidence. Falling back to it is what taught
        # the served model to call documented human code generated, so a build that would
        # do it again stops here instead.
        raise SystemExit(
            "[!] --repository-labels is required with --repos: a real repository's target "
            "is only trustworthy when a verified authorship label and its measured "
            "attribution decide it. Run `uv run src/cli.py harvest --labels-out "
            "data/repository_labels.jsonl` first, or set HACKGRADER_SNIPPET_ONLY=1 to build "
            "without repository rows."
        )
    fetcher = repo_fetcher() or (lambda _url: "")
    return repo_rows(path, fetcher, labels=labels)


def _require_independent_repositories(rows: list[dict[str, Any]]) -> None:
    """Refuse a pool that cannot give every split its own real repositories."""
    independent_urls = {
        canonical_source_url(row["source_url"])
        for row in rows
        if row["source_type"] == "real_repository"
    }
    required = 3 * MIN_REAL_REPOSITORIES_PER_SPLIT
    if len(independent_urls) < required:
        raise SystemExit(
            f"[!] Only {len(independent_urls)} independent real repositories were "
            f"generated; at least {required} are needed "
            f"({MIN_REAL_REPOSITORIES_PER_SPLIT} per source-disjoint split)."
        )


def _validate_candidates(rows: list[dict[str, Any]]) -> None:
    """Reject any candidate row the live data contract would refuse at training time."""
    for index, row in enumerate(rows):
        try:
            validate_row(row, "candidate dataset", index, report)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"[!] Candidate row failed the live data contract: {exc}"
            ) from None


def build_candidate_rows(
    args,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Construct all examples, finalize runtime-shaped rows, and filter invalid targets."""
    manifest = repository_manifest(getattr(args, "repository_labels", None))
    if manifest is not None:
        report_manifest(manifest)
        require_usable(manifest)
    samples = aig_samples(args.rows)
    snippets = snippet_rows(samples)
    synthetic = synthetic_repo_rows(samples, SYNTHETIC_REPOS)
    artifacts = artifact_rows(samples, ARTIFACT_ROWS)
    repositories = real_repositories(args.repos, manifest)
    weights = feature_weights(snippets + synthetic + artifacts)
    try:
        written = finalize(
            snippets + repositories + synthetic + artifacts,
            weights,
            tokenizer=candidate_tokenizer(),
            max_tokens=DEFAULT_MAX_LEN,
        )
    except ValueError as exc:
        raise SystemExit(
            f"[!] Could not fit repository evidence to the training window: {exc}"
        ) from None
    rows = [row for row in written if usable(row)]
    if not rows:
        raise SystemExit(
            "[!] No valid examples were produced; check dataset access, repository reads, "
            "and configured generation quotas."
        )
    _require_independent_repositories(rows)
    _validate_candidates(rows)
    return rows, weights
