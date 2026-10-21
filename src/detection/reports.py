"""Generate judge-shaped reports from deterministic detector measurements.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import random

from contracts import render_report

from .analysis import analyze
from .base import EXPLICIT_ARTIFACT_WEIGHT
from .metrics import files_in, identifiers


def build_summary(
    code: str,
    ai_score: int,
    ai_evidence: list[str],
    rng: random.Random,
) -> str:
    """Name the files and the strongest signal in one short sentence.

    The target has to match the instruction the model is served: one sentence of at most 25
    words. A three-sentence target taught the model to write a paragraph that the report
    then had to cut.
    """
    files = files_in(code)
    names = identifiers(code)
    subjects = [f"`{name}`" for name in (files[:2] + names[:2])] or ["the submitted code"]
    subject = " and ".join(subjects[:2])
    lead = (
        ai_evidence[0] if ai_evidence else "no generated-code markers were found"
    ).split(" - ")[0].strip()
    openers = (
        f"The strongest signals here sit in {subject}",
        f"The clearest evidence sits in {subject}",
        f"This score rests on {subject}",
    )
    return f"{rng.choice(openers)}: the judge's AI-involvement score is {ai_score}% based on {lead}."


def _measures_nothing(entry: str) -> bool:
    """Return whether an evidence entry only states that no marker was found."""
    return not entry or entry.lower().startswith(
        ("no generated-code markers", "no ai-specific")
    )


def _head(entry: str) -> str:
    """Return the measurement an evidence entry rests on, without its quote or file."""
    if _measures_nothing(entry):
        return ""
    return entry.split(" - ", 1)[0].split(", for example", 1)[0].strip().rstrip(".")


def _ai_reason(
    ai_score: int, ai_evidence: list[str], explicit_artifact: bool
) -> str:
    """State what sets the AI-involvement number and what would move it either way.

    The line answers the question the prompt asks of the judge — which code sets this
    number, and what would move it — rather than restating the score. It also fixes the
    direction of travel: a quoted generation marker moves the score up, and a view with
    less of the structure it currently rests on moves it down.
    """
    if explicit_artifact:
        return (
            "The generation marker quoted under Rank 1 is the sign that sets this "
            "number: no style measurement in the view outweighs a line stating the code "
            "was generated. A batch carrying no such marker would score lower."
        )
    head = _head(ai_evidence[0]) if ai_evidence else ""
    if head:
        return (
            f"Rank 1 sets this number: {head}. That is structure the view shows rather "
            "than a quoted generator marker, so a generation banner would move the score "
            "up and less of that structure would move it down."
        )
    return (
        "The view offers no measured authorship sign to rest on. A quoted generation "
        "banner would move the score up; a batch with even less structure would move it "
        "down."
    )


def _quality_reason(deductions: list[str]) -> str:
    """State which deductions set the quality number and what would move it."""
    if deductions:
        return (
            f"The deductions listed here set this number, starting with {deductions[0]}. "
            "Removing that problem would raise it; a long unbroken function or a "
            "missing README would lower it."
        )
    return (
        "No deduction applies to the files shown, so nothing lowers this number. A long "
        "unbroken function or a missing README would lower it."
    )


def build_report(
    code: str,
    seed: int = 0,
    ai_score: int | None = None,
    ai_evidence: list[str] | None = None,
) -> str:
    """Build a deterministic training example in the same format requested from the judge."""
    rng = random.Random(seed)
    result = analyze(code)
    ai_score = result["ai_score"] if ai_score is None else ai_score
    if ai_evidence is None:
        ai_evidence = result["ai_evidence"] or [
            "no generated-code markers were found"
        ]
        if result["human_evidence"]:
            ai_evidence += [f"human traits present: {result['human_evidence'][0]}" ]
    explicit = any(
        (finding.get("weight") or 0) >= EXPLICIT_ARTIFACT_WEIGHT
        for finding in result["ai_findings"]
    )
    return render_report(
        ai_score=ai_score,
        ai_evidence=ai_evidence,
        quality_score=result["quality_score"],
        quality_deductions=result["quality_deductions"],
        summary=build_summary(code, ai_score, ai_evidence, rng),
        ai_reason=_ai_reason(ai_score, ai_evidence, explicit),
        quality_reason=_quality_reason(result["quality_deductions"]),
    )
