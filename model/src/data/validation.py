"""Validate chat rows against provenance and the live report contract.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import unquote, urlsplit

from .config import SNIPPET_ONLY

DEFAULT_MAX_LEN = int(os.environ.get("HACKGRADER_MAX_LEN", "3072"))
MAX_TARGET_SHARE = 0.30
EXPECTED_SOURCE_TYPES = frozenset(
    {"aigcodeset", "synthetic_repository", "artifact", "real_repository"}
)
SOURCE_TYPES = EXPECTED_SOURCE_TYPES


def required_source_types(snippet_only: bool) -> frozenset[str]:
    """Return the source families a complete build must contain.

    ``EXPECTED_SOURCE_TYPES`` is what a row may declare; this is what a build must supply.
    The two differ only in snippet-only mode, where every stage stops requiring the
    real-repository family rather than each check inventing its own exception.
    """
    if snippet_only:
        return EXPECTED_SOURCE_TYPES - {"real_repository"}
    return EXPECTED_SOURCE_TYPES


REQUIRED_SOURCE_TYPES = required_source_types(SNIPPET_ONLY)
REQUIRED_SPLIT_SOURCE_TYPES = REQUIRED_SOURCE_TYPES - {
    "artifact",
    "synthetic_repository",
}
MESSAGE_ROLES = ("system", "user", "assistant")
CONTENT_BEGIN = "===== BEGIN REPOSITORY CONTENT =====\n"
CONTENT_END = "\n===== END REPOSITORY CONTENT ====="
_SUSPICIOUS_LABEL_SLUG = re.compile(
    r"(?:^|[-_])(?:ai|human|llm|machine|generated|written|label|label-[01])(?:$|[-_])",
    re.IGNORECASE,
)


def canonical_source_url(value: str) -> str:
    """Normalize GitHub aliases, suffixes, and deep links to one repository identity."""
    candidate = value.strip()
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = urlsplit(candidate)
    hostname = (parsed.hostname or "").casefold()
    if hostname in {"github.com", "www.github.com"}:
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) >= 2:
            return f"github.com/{parts[0]}/{parts[1].removesuffix('.git')}".casefold()
    return value.strip().rstrip("/").removesuffix(".git").casefold()


def slug_exposes_label(source_url: str) -> bool:
    """Return whether a URL's slug carries a word that reveals the class.

    A repository called ``project-AI`` hands the model its own label in the prompt, which
    teaches it to read the URL instead of the code. The check is public so a builder can
    drop those repositories from its pool before a build reaches the row contract.
    """
    candidate = source_url if "://" in source_url else f"https://{source_url}"
    return _SUSPICIOUS_LABEL_SLUG.search(unquote(urlsplit(candidate).path)) is not None


def _validate_unlabeled_source_url(source_url: str, path: str, index: int) -> None:
    """Reject source URL slugs containing likely ground-truth class indicators."""
    candidate = source_url if "://" in source_url else f"https://{source_url}"
    parsed = urlsplit(candidate)
    decoded_path = unquote(parsed.path)
    if _SUSPICIOUS_LABEL_SLUG.search(decoded_path):
        raise ValueError(
            f"{path} row {index} has a source URL that exposes a likely class label"
        )
    username = parsed.username or ""
    password = parsed.password or ""
    if username or password:
        raise ValueError(f"{path} row {index} source URL may not include credentials")


def validate_row(
    row: dict[str, Any], path: str, index: int, report_contract: Any
) -> dict[str, Any]:
    """Check message roles, prompt parity, supported citations, and source metadata."""
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != len(MESSAGE_ROLES):
        raise ValueError(f"{path} row {index} must contain system, user, and assistant messages")
    for position, (message, expected_role) in enumerate(zip(messages, MESSAGE_ROLES, strict=True)):
        if not isinstance(message, dict) or message.get("role") != expected_role:
            raise ValueError(f"{path} row {index} message {position} must have role '{expected_role}'")
        if not isinstance(message.get("content"), str):
            raise TypeError(f"{path} row {index} message {position} must have string content")

    system, user, assistant = messages
    if system["content"] != report_contract.SYSTEM_PROMPT:
        raise ValueError(f"{path} row {index} uses a system prompt different from live grading")
    user_text = user["content"]
    if user_text.count(CONTENT_BEGIN) != 1 or user_text.count(CONTENT_END) != 1:
        raise ValueError(f"{path} row {index} must delimit exactly one repository evidence view")
    if user_text.index(CONTENT_BEGIN) >= user_text.index(CONTENT_END):
        raise ValueError(f"{path} row {index} has reversed repository-content delimiters")
    if not user_text.rstrip().endswith(report_contract.REPORT_INSTRUCTION):
        raise ValueError(f"{path} row {index} does not end with the live report instruction")
    if "repository evidence batch " not in user_text:
        raise ValueError(f"{path} row {index} is missing its bounded evidence-batch question")

    parsed = report_contract.parse_report(assistant["content"])
    if not parsed["valid"] or not parsed["citations"]:
        raise ValueError(f"{path} row {index} has an invalid or uncited assistant target")
    evidence = user_text.split(CONTENT_BEGIN, 1)[1].split(CONTENT_END, 1)[0]
    unsupported = report_contract.verify_citations(assistant["content"], evidence)
    if unsupported:
        raise ValueError(f"{path} row {index} target cites evidence missing from its view: {unsupported[0]}")

    source_url = row.get("source_url")
    source_type = row.get("source_type")
    source_kind = row.get("source_kind")
    groups = row.get("source_groups")
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError(f"{path} row {index} has no source_url provenance")
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"{path} row {index} has an unknown source_type: {source_type!r}")
    if source_kind not in {"snippet", "repository"}:
        raise ValueError(f"{path} row {index} has an unknown source_kind: {source_kind!r}")
    if not isinstance(groups, list) or not groups or any(not isinstance(item, str) or not item.strip() for item in groups):
        raise ValueError(f"{path} row {index} has no usable source_groups provenance")
    if source_type == "aigcodeset" and source_kind != "snippet":
        raise ValueError(f"{path} row {index} mislabels an AIGCodeSet sample as a repository")
    if source_type == "real_repository" and source_kind != "repository":
        raise ValueError(f"{path} row {index} mislabels a real repository as a snippet")
    _validate_unlabeled_source_url(source_url, path, index)
    if source_url not in user_text:
        raise ValueError(f"{path} row {index} source_url is not named in its user prompt")

    parsed_url = urlsplit(source_url if "://" in source_url else f"https://{source_url}")
    path_parts = [part for part in parsed_url.path.split("/") if part]
    if source_type == "real_repository" and (
        parsed_url.hostname is None
        or parsed_url.hostname.casefold() not in {"github.com", "www.github.com"}
        or len(path_parts) < 2
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise ValueError(f"{path} row {index} real_repository source is not a GitHub URL")
    return row
