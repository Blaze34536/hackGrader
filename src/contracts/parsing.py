"""Format, parse, and validate the grader's three-part report.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import re
from typing import Any

from .prompts import (
    AI_EVIDENCE_PATTERN,
    DEDUCTION_PATTERN,
    EVIDENCE_ITEM_PATTERN,
    FILE_NAME_HEADER,
    MIN_SUMMARY_CHARS,
    NON_EVIDENCE_PREFIXES,
    PLACEHOLDER_MARKERS,
    QUOTED_TOKEN,
    SCORE_PATTERNS,
    SUMMARY_PATTERN,
    WHY_SCORE_LINE,
)
from .repository_dump import (
    DOCUMENTATION_EXTENSIONS,
    SOURCE_EXTENSIONS,
    carries_authorship_evidence,
)

RANKED_EVIDENCE_LINE_PATTERN = re.compile(
    r"^\s*Rank\s*(\d+)[.):]\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE
)

# Wording that states an impression or an absence rather than a measured line of the view.
# The prompt forbids it, and an entry that uses it is not evidence even when it also quotes a
# real line: "the code reads as machine-written, for example `#Title` in app.py" quotes a
# heading that says nothing about who wrote it.
IMPRESSION_CLAIM = re.compile(
    r"(?i)\b(reads? as|looks? (like|machine|hand)|appears? to be|seems? to be"
    r"|machine[- ]written|hand[- ]written|human[- ]typed"
    r"|no human|zero human|0 human)\b"
)

# Style counts the contract does not publish. The measurement block carries no indentation
# or whitespace figure and the prompt states that formatting is not authorship evidence, so
# an entry whose reason rests on one is claiming a measurement the view never showed. That
# claim was the served judge's strongest sign on ordinary projects, quoted against a real
# line so the quote check alone could not catch it: "0 line(s) are tab indented, for example
# `using UnityEngine;` in CatClicked.cs" scored 77% on a stored run.
STYLE_CLAIM = re.compile(
    r"(?i)\b(tab[- ]indented|indentation(?: width| widths| style| discipline)?|"
    r"trailing whitespace|line width|whitespace)\b"
)

# Report text that states no claim at all. It is neither evidence nor an unsupported
# citation, so it is dropped before either is decided.
NO_CLAIM_SENTINELS = (
    "n/a",
    "none",
    "no deductions applied",
    "no generated-code markers were found",
    "no ai-specific evidence found",
    "no ai-specific signs were found",
)
# The extension is read in full: a cap on its length cut ``Main.storyboard`` to ``Main.storyb``,
# which the view then reported as a file it did not reproduce.
CITED_NAME = re.compile(r"(?i)\bin\s+`?([A-Za-z0-9_./-]+\.[A-Za-z0-9]+)`?")
CODE_FILE_SUFFIXES = SOURCE_EXTENSIONS + DOCUMENTATION_EXTENSIONS + (
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".cfg",
    ".ini",
    ".env",
)


def render_report(
    ai_score: int,
    ai_evidence: list[str],
    quality_score: int,
    quality_deductions: list[str],
    summary: str,
) -> str:
    """Render a structured report in the shared judge report format."""
    ranked = ai_evidence[:8]
    evidence = "\n".join(
        f"Rank {index}: {entry}" for index, entry in enumerate(ranked, start=1)
    ) or "No AI-specific evidence found."
    deductions = "; ".join(quality_deductions[:4]) or "no deductions applied"
    return (
        f"1. AI Involvement Score (0-100%): {ai_score}%\n"
        f"Ranked AI evidence (strongest to weakest):\n{evidence}\n\n"
        f"2. Code Quality Score (0-100%): {quality_score}%\n"
        f"- Deductions: {deductions}\n\n"
        f"3. Summary & Notable Findings: {summary}"
    )


def _entries(matches) -> list[str]:
    """Split semi-colon separated report entries out of a pattern's matches."""
    entries: list[str] = []
    for match in matches:
        for entry in match.group(1).split("; "):
            entry = entry.strip().strip(".")
            if entry and entry.lower() not in NO_CLAIM_SENTINELS:
                entries.append(entry)
    return entries


def _ranked_evidence_is_valid(evidence: str) -> bool:
    """Require consecutive ranks starting at one, or an explicit empty-evidence note.

    A score justification is not an evidence entry. The report asks for each justification
    directly under its score, where it falls outside this block, but a report that places one
    after the ranks is still complete, so the line is skipped rather than read as a malformed
    entry.
    """
    lines = [
        line.strip()
        for line in evidence.splitlines()
        if line.strip() and not WHY_SCORE_LINE.match(line.strip())
    ]
    if len(lines) == 1 and lines[0].rstrip(".").casefold() in {
        "no ai-specific evidence found",
        "no ai-specific signs were found",
    }:
        return True
    entries = [RANKED_EVIDENCE_LINE_PATTERN.fullmatch(line) for line in lines]
    if not entries or any(entry is None for entry in entries):
        return False
    ranks = [int(entry.group(1)) for entry in entries if entry is not None]
    return ranks == list(range(1, len(ranks) + 1))


def evidence_claims(content: str) -> list[str]:
    """Return judge-ranked AI evidence entries, strongest first.

    Numbering is removed from each entry for citation validation and PDF readability.
    Legacy single-line ``Evidence:`` reports remain readable.
    """
    text = content or ""
    match = AI_EVIDENCE_PATTERN.search(text)
    if not match:
        return []
    evidence = match.group(1).strip()
    ranked = [item.strip() for item in EVIDENCE_ITEM_PATTERN.findall(evidence)]
    if ranked:
        return [entry for entry in ranked if entry.lower() not in NO_CLAIM_SENTINELS]
    return [
        entry.strip()
        for entry in evidence.split("; ")
        if entry.strip() and entry.strip().lower() not in NO_CLAIM_SENTINELS
    ]


def deduction_claims(content: str) -> list[str]:
    """Return the Deductions entries: the score adjustments a report applied.

    Deductions price the work rather than describe the repository, so they are exempt from
    citation verification and are read back separately from the evidence that must quote code.
    """
    return _entries(DEDUCTION_PATTERN.finditer(content or ""))


def extract_evidence(content: str) -> list[str]:
    """Extract ranked AI evidence and deductions from a report."""
    return evidence_claims(content) + deduction_claims(content)


def is_bookkeeping(quote: str) -> bool:
    """Return whether quoted text describes the view rather than the repository."""
    text = quote.strip().lstrip("-* \t").strip()
    return text.startswith("[...") or text.startswith("--- FILE:") or any(
        text.startswith(prefix) for prefix in NON_EVIDENCE_PREFIXES
    )


def reproduced_files(repo_block: str) -> set[str]:
    """Return the paths and basenames a view actually reproduces for the model."""
    names = {
        match.group(1).strip() for match in FILE_NAME_HEADER.finditer(repo_block or "")
    }
    return names | {name.rsplit("/", 1)[-1] for name in names}


def file_sections(repo_block: str) -> dict[str, str]:
    """Map each reproduced path and basename to the text the view shows for that file."""
    text = repo_block or ""
    headers = list(FILE_NAME_HEADER.finditer(text))
    sections: dict[str, str] = {}
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        name = header.group(1).strip()
        body = text[header.end() : end]
        for key in (name, name.rsplit("/", 1)[-1]):
            sections[key] = f"{sections[key]}\n{body}" if key in sections else body
    return sections


def _names_a_file(token: str) -> bool:
    r"""Return whether a quoted token is a path a citation could be citing.

    Quoted repository content routinely contains a slash -- an `src="songs/theme.mp3"`
    attribute, a route, a `data:image/jpeg` prefix -- and reading every such quote as a cited
    path rejected evidence the view had really reproduced. A quoted file name is a single
    path-shaped token, so anything with whitespace or punctuation around it is content.
    """
    token = token.strip()
    if not token or any(char.isspace() for char in token):
        return False
    if not re.fullmatch(r"[A-Za-z0-9_./\\-]+", token):
        return False
    # A repository path is relative and has no empty segment. A quoted comment marker such as
    # ``//asq_start`` or ``/* setup */`` has a leading or doubled slash, so it is content.
    if token.startswith("/") or "//" in token or token.endswith("/"):
        return False
    # A slash alone does not make a file: a date or a fragment such as ``9/02/22`` has one too.
    # A cited file ends in a recognised extension, so only that shape is checked against the view.
    return token.lower().endswith(CODE_FILE_SUFFIXES)


def cited_files(citation: str) -> list[str]:
    """Return the file names a citation claims to have read.

    Quoted spans are removed before the prose is scanned: a quote is repository content, and
    English inside code says "in" constantly. Reading `for line in os.environ.get(...)` as a
    cited path rejected a real quote from a file the view had reproduced, because
    ``os.environ.get`` looks exactly like a name with an extension.
    """
    prose = QUOTED_TOKEN.sub("", citation or "")
    names = [match.group(1) for match in CITED_NAME.finditer(prose)]
    names += [
        token
        for token in QUOTED_TOKEN.findall(citation or "")
        if _names_a_file(token)
    ]
    return list(dict.fromkeys(names))


def cites_only_packaging(citation: str) -> bool:
    """Return whether every file an entry names packages or describes the project.

    An entry that names no file is not judged here: a view without file headers cannot be
    classified, so its quote is checked like any other. A README, a manifest or lock file, a
    licence, and build configuration are what every project of that shape carries, so a sign
    read from one of them says how the project is packaged rather than who wrote the code.
    """
    named = cited_files(citation)
    return bool(named) and not any(carries_authorship_evidence(name) for name in named)


def cites_formatting(citation: str) -> bool:
    """Return whether an entry rests on a style count the view does not publish."""
    return bool(STYLE_CLAIM.search(QUOTED_TOKEN.sub("", citation or "")))


def is_authorship_evidence(citation: str) -> bool:
    """Return whether an entry may stand as a sign about who wrote the code."""
    return not cites_only_packaging(citation) and not cites_formatting(citation)


def authorship_evidence(content: str) -> list[str]:
    """Return the ranked evidence entries that rest on source code.

    Both score consumers read evidence through here, so a sign the prompt forbids cannot be
    published: the aggregate report and the stored annotation keep only entries whose named
    files include code and whose reason rests on something other than formatting, while
    ``verify_citations`` reports the rest as unsupported for review. Filtering rather than
    reordering keeps the judge's own strongest-first order among the entries that survive.
    """
    return [entry for entry in evidence_claims(content) if is_authorship_evidence(entry)]


def parse_report(content: str) -> dict[str, Any]:
    """Parse scores, ranked citations, summary, and validity from a model response."""
    text = content or ""
    scores: dict[str, int | None] = {}
    for key, pattern in SCORE_PATTERNS.items():
        match = pattern.search(text)
        value = None
        if match:
            parsed = int(match.group(1))
            if 0 <= parsed <= 100:
                value = parsed
        scores[key] = value

    summary_match = SUMMARY_PATTERN.search(text)
    summary = summary_match.group(1).strip() if summary_match else ""
    return {
        "ai_score": scores["ai_score"],
        "code_score": scores["code_score"],
        "summary": summary,
        "citations": extract_evidence(text),
        "valid": is_valid_report(text),
    }


def is_valid_report(content: str) -> bool:
    """Return whether a response contains real scores and a non-placeholder summary."""
    text = (content or "").strip()
    if not text or "Error during chat" in text or "llama-server" in text:
        return False

    lowered = text.lower()
    if any(marker in lowered for marker in PLACEHOLDER_MARKERS):
        return False
    if "{" in text and '"name"' in text and '"arguments"' in text:
        return False

    for pattern in SCORE_PATTERNS.values():
        match = pattern.search(text)
        if not match or not 0 <= int(match.group(1)) <= 100:
            return False

    summary_match = SUMMARY_PATTERN.search(text)
    if not summary_match or len(summary_match.group(1).strip()) < MIN_SUMMARY_CHARS:
        return False
    evidence_match = AI_EVIDENCE_PATTERN.search(text)
    if not evidence_match:
        return False
    evidence = evidence_match.group(1)
    return _ranked_evidence_is_valid(evidence)


def is_boilerplate_report(content: str) -> bool:
    """Identify a score-shaped answer that repeats one of the known placeholders."""
    text = (content or "").strip()
    if not text or not all(pattern.search(text) for pattern in SCORE_PATTERNS.values()):
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def verify_citations(content: str, repo_block: str) -> list[str]:
    """Return evidence entries whose quotes or file names the view does not support.

    Every quoted token must occur in the view, any file an entry names must be one the view
    reproduces, and the quotes must occur in the file the entry names. Accepting an entry when
    *any* one of its tokens appeared let a real quote carry a fabricated file name, and --
    because bookkeeping lines are strings in the view too -- let a coverage header stand in for
    evidence. Checking only the whole view let a real line from one file be cited as another.
    """
    block = repo_block or ""
    files = reproduced_files(block)
    sections = file_sections(block)
    unsupported: list[str] = []
    for citation in evidence_claims(content):
        quotes = [
            token.strip() for token in QUOTED_TOKEN.findall(citation) if token.strip()
        ]
        if not quotes:
            unsupported.append(f"{citation} - quotes no text from the repository view")
            continue
        if all(is_bookkeeping(quote) for quote in quotes):
            unsupported.append(
                f"{citation} - quotes repository bookkeeping, not file content"
            )
            continue
        if cites_only_packaging(citation):
            unsupported.append(
                f"{citation} - cites the project's description or tooling, not its code"
            )
            continue
        if IMPRESSION_CLAIM.search(QUOTED_TOKEN.sub("", citation)):
            unsupported.append(
                f"{citation} - states an impression or an absence, not a measured line"
            )
            continue
        if cites_formatting(citation):
            unsupported.append(
                f"{citation} - rests on formatting, which the view does not measure"
            )
            continue
        missing = [quote for quote in quotes if quote not in block]
        if missing:
            unsupported.append(
                f"{citation} - quotes text absent from the view: {missing[0][:80]}"
            )
            continue
        cited = cited_files(citation)
        unknown = [name for name in cited if name not in files]
        if unknown:
            unsupported.append(
                f"{citation} - names a file this view does not reproduce: {unknown[0]}"
            )
            continue
        if cited:
            # A quote found in some other file of the view is not evidence about the one
            # the entry names, so the quoted text must appear in the named file itself.
            named = "\n".join(sections[name] for name in cited)
            absent = [quote for quote in quotes if quote not in named]
            if absent:
                unsupported.append(
                    f"{citation} - quotes text absent from the file it names: "
                    f"{absent[0][:80]}"
                )
    return unsupported
