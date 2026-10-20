"""Generate judge-shaped reports from deterministic detector measurements.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import random

from contracts import render_report

from .analysis import analyze
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
    return render_report(
        ai_score=ai_score,
        ai_evidence=ai_evidence,
        quality_score=result["quality_score"],
        quality_deductions=result["quality_deductions"],
        summary=build_summary(code, ai_score, ai_evidence, rng),
    )
