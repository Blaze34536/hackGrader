"""Attach source-grounded evidence to target examples.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import re
from typing import Any

import detection as detector
from contracts.parsing import carries_authorship_evidence, cited_files, is_bookkeeping

from .config import DOCUMENTATION_FEATURES
from .features import feature_supported, has_explicit_artifact, machine_likeness

# A verified label does not rest on the detector's style separation, so weights fitted to
# that separation cannot decide whether its citation exists. Passing those weights dropped
# most verified repository batches for having no README to quote in a code-only view — the
# same style gate one layer down, which left a build with 80 verified repositories and too
# few rows to split. A verified row is cited from code measurements alone.
VERIFIED_EVIDENCE_WEIGHTS = {
    "documented functions": 0.4,
    "comment density": 0.4,
    "human traces": 0.2,
}

# A rendered evidence view opens with coverage/provenance headers (``REPOSITORY:``,
# ``FILES FOUND:``, ``FILES INCLUDED (n):``, ``FILES TRUNCATED:``) and closes with a
# measurements block. None of those lines is repository content, so citing one as
# authorship evidence would attach a meaningless quote to a target and teach the model
# that bookkeeping headers are findings.
_SKIPPED_PREFIXES = (
    "#",
    "//",
    "/*",
    "*",
    "REPOSITORY:",
    "DEFAULT BRANCH:",
    "FILES ",
    "MEASUREMENTS TAKEN FROM",
)
_HEADER_PREFIXES = (
    "REPOSITORY:",
    "DEFAULT BRANCH:",
    "FILES ",
    "--- FILE:",
    "[... ",
)
# An omission marker (a truncated excerpt, an omitted file, a redacted encoded payload) is
# bookkeeping about the view. A line carrying one is not repository content, so quoting it
# would attach a target to text the serving citation check refuses - and a long encoded
# payload is exactly the line a selector reaches for when it wants the longest line in a
# stylesheet.
_OMISSION_MARKER = "[... "
_DOCSTRING_LINE = re.compile(r"^\s*(?:[rbu]{0,2}(?:\"\"\"|'''))")
_COMMENT_LINE = re.compile(r"^\s*(?:#|//|/\*|\*)")


def _dump_lines(prompt_code: str) -> list[tuple[str, str]]:
    """Pair each repository content line with its enclosing filename.

    A view that marks no ``--- FILE:`` header names no file, and it stays that way here: a
    stand-in name would teach the model to name a file the view never showed it, which is
    exactly the fabricated citation the serving guard rejects.
    """
    name = ""
    lines = []
    for raw in prompt_code.splitlines():
        stripped = raw.strip()
        if stripped.startswith("--- FILE: ") and stripped.endswith("---"):
            name = stripped[len("--- FILE: ") : -len(" ---")]
            continue
        if stripped.startswith(_HEADER_PREFIXES) or _OMISSION_MARKER in stripped:
            continue
        if is_bookkeeping(stripped):
            continue
        lines.append((raw, name))
    return lines


def _quotable(line: str) -> str:
    """Return the leading span of a source line that one backtick quote can carry.

    An evidence entry wraps its quote in backticks, and the report parser splits entries on
    ``; ``. A quote holding either one is cut short, so the citation loses its closing
    backtick and the serving guard rejects it. Cutting at those separators keeps the quote a
    verbatim substring of the view.
    """
    head = line.split("; ", 1)[0].split("`", 1)[0].strip()[:60]
    # A span of punctuation alone (a ``//-----`` divider) measures nothing a reader can
    # check, and the citation check reads it as a file path, so it is not quotable.
    if not any(char.isalnum() for char in head):
        return ""
    # A line that is itself a path (``notes/TODO_list.txt``) is read by the citation check as a
    # file the view must show as a header. It is not a header, so the quote would be rejected.
    return "" if cited_files(f"`{head}`") else head


def _authorship_lines(prompt_code: str) -> list[tuple[str, str]]:
    """Return only the view's lines whose file can say who wrote the project.

    A target that quotes the README, a manifest, or a configuration file teaches the model
    that packaging is a sign, which is the behavior the served grader reproduced with "the
    README has 7 heading(s) and 2 bullet(s)" for an 81% score. Only code files may supply
    the quote. A view with no file headers names no file, so its lines stay eligible and
    the citation check judges them on their text alone.
    """
    return [
        (text, name)
        for text, name in _dump_lines(prompt_code)
        if not name or carries_authorship_evidence(name)
    ]


def signal_line(prompt_code: str, label: int, patterns: tuple = ()) -> tuple[str, str] | None:
    """Select a code line that directly supports an example label."""
    lines = _authorship_lines(prompt_code)

    def first(selector):
        """Find the first repository line matching a signal predicate that yields a quotable span."""
        for text, name in lines:
            stripped = text.strip()
            if stripped and selector(stripped):
                quote = _quotable(stripped)
                if quote:
                    return quote, name
        return None

    for pattern in patterns:
        found = first(lambda text, pattern=pattern: pattern.search(text) is not None)
        if found:
            return found
    # A human-labeled example falls straight through to its first content line: indentation
    # is not evidence of a person either, and picking an indented line as the quote said so.
    for pattern in (_DOCSTRING_LINE, _COMMENT_LINE) if label else ():
        found = first(lambda text, pattern=pattern: pattern.match(text) is not None)
        if found:
            return found
    return first(lambda text: not text.startswith(_SKIPPED_PREFIXES))


def explicit_patterns() -> tuple[re.Pattern, ...]:
    """Return the detector regexes that identify explicit authorship artifacts."""
    return tuple(
        signal[0] for signal in detector.AI_SIGNALS
        if signal[1] >= detector.EXPLICIT_ARTIFACT_WEIGHT
    )


def _absent_traces(metrics: dict[str, Any], name: str) -> bool:
    """Return whether a trace feature counts zero traces, which is an absence, not evidence.

    "0 human trace(s) appear in the source" states what the source lacks, and the prompt
    forbids that claim, so a target never cites the trace feature when it found none.
    """
    if name != "human traces":
        return False
    return sum(metrics.get(trace) or 0 for trace in detector.TRACES) == 0


def evidence_phrase(metrics: dict[str, Any], weights: dict[str, float], label: int) -> str:
    """Describe the strongest readable feature supporting a labeled target.

    Features whose supporting evidence is absent are skipped, so a target never justifies a
    score with a measurement over text the example does not contain: that reading is not
    evidence a reader could check, and it teaches a vacuous justification.
    """
    if not weights:
        return ""
    readable = [
        name for name in weights
        if feature_supported(metrics, name)
        and not _absent_traces(metrics, name)
    ]
    if not readable:
        return ""
    values = machine_likeness(metrics)
    # A code feature heads every target. The documentation features describe the project
    # rather than the code, so one of them leads the phrase only when no code measurement in
    # this example is readable at all - otherwise a target justifies its score with the
    # README while the code that decided it goes unmentioned.
    code_readable = [name for name in readable if name not in DOCUMENTATION_FEATURES]
    key = max(
        code_readable or readable,
        key=lambda name: weights[name] * (values[name] if label else 1 - values[name]),
    )
    phrases = {
        "readme structure": f"the README has {metrics.get('readme_headings', 0)} heading(s) and {metrics.get('readme_bullets', 0)} bullet(s)",
        "readme first person": f"the README has {metrics.get('readme_first_person', 0)} first-person reference(s)",
        "readme informal": f"the README has {(metrics.get('readme_casual') or 0) + (metrics.get('readme_typos') or 0)} informal marker(s)",
        "documented functions": f"{metrics.get('docstrings', 0)} of {metrics.get('functions', 0)} definitions have docstrings",
        "restated docstrings": f"{metrics.get('restated_docstrings', 0)} of {metrics.get('docstrings', 0)} docstrings restate a name",
        "comment density": f"{metrics.get('comment_lines', 0)} of {metrics.get('code_lines', 0)} source lines are comments",
        "human traces": f"{sum(metrics.get(name, 0) or 0 for name in detector.TRACES)} human trace(s) appear in the source",
    }
    return phrases[key]


def _evidence_sentence(head: str, citation: tuple[str, str]) -> str:
    """Attach a quote to its head, naming the source file only when the view has one."""
    line, name = citation
    where = f" in {name}" if name else ""
    return f"{head}, for example `{line}`{where}"


def label_evidence(
    prompt_code: str,
    label: int,
    analysis: dict[str, Any],
    weights: dict[str, float],
) -> list[str]:
    """Explain a target label with the measurement it rests on and a file-qualified quote.

    The entry states what was measured, never which author wrote the code. Opening every
    target with "the code reads as machine-written" whenever the dataset label says so
    teaches the model to assert an authorship verdict the view does not support, and it is
    the claim nothing in the view can check: served against real repositories, that sentence
    was reproduced for code scoring 2% and attached to whatever file happened to be quoted.
    Only a quoted AI-generation marker names machine authorship here, because only that
    quote says so itself.

    A batch whose view offers nothing to quote, or whose measurements support no readable
    phrase, yields no evidence at all: the caller drops it. Supervising a target that
    quotes nothing, or names a file the view never showed, teaches a citation the serving
    guard then rejects.
    """
    if label and has_explicit_artifact(analysis):
        citation = signal_line(prompt_code, label, explicit_patterns())
        if citation:
            return [
                _evidence_sentence(
                    "the code carries an explicit AI-generation marker", citation
                )
            ]
    phrase = evidence_phrase(analysis["metrics"], weights, label)
    if not phrase:
        return []
    patterns = explicit_patterns() if label else tuple(
        signal[0] for signal in detector.HUMAN_SIGNALS
    )
    citation = signal_line(prompt_code, label, patterns)
    if not citation:
        return []
    return [_evidence_sentence(phrase, citation)]
