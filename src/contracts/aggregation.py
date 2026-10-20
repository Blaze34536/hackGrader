"""Combine per-view judge reports into one repository-level response.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from typing import Any

from .parsing import authorship_evidence, deduction_claims
from .prompts import SUMMARY_PATTERN


def combine_reports(
    reports: list[dict[str, Any]],
    source_line_weights: list[int],
) -> dict[str, Any]:
    """Line-weight the batches that validated and record the ones that did not.

    A single unusable view must not discard a repository. A batch the model answered with no
    report is dropped from every aggregate - scores, citations, and deductions - and listed
    under ``failed_batches`` so the omission stays visible. Only a repository whose every
    batch failed is a failure in its own right.

    The aggregate is the line-weighted mean of the batches that validated, and nothing
    replaces it. Citations the view does not support remain available in
    ``unsupported_citations`` for inspection. Published evidence is the strongest the
    batches offered: entries that name only the project's description or tooling are
    dropped, and the rest rank by the batch whose score they carried, because the batch
    that scored highest is the one whose evidence decided the repository's number.
    """
    if not reports:
        return {"status": 502, "content": "No repository reports were produced."}

    kept = [
        (item, max(weight, 1))
        for item, weight in zip(reports, source_line_weights, strict=True)
        if item["parsed"]["valid"]
    ]
    failed = [
        {"index": index, "content": item["content"]}
        for index, item in enumerate(reports)
        if not item["parsed"]["valid"]
    ]
    if not kept:
        return {
            "status": 502,
            "content": (
                f"None of the {len(reports)} repository evidence batches produced a valid "
                "report. No aggregate score was accepted. "
                f"First response: {reports[0]['content'][:240]}"
            ),
            "batch_count": len(reports),
            "batch_reports": [item["content"] for item in reports],
            "failed_batches": failed,
        }

    total_weight = sum(weight for _, weight in kept)
    ai_score = round(
        sum(item["parsed"]["ai_score"] * weight for item, weight in kept) / total_weight
    )
    code_score = round(
        sum(item["parsed"]["code_score"] * weight for item, weight in kept) / total_weight
    )

    ai_citations: list[str] = []
    ranked: list[tuple[int, int, str]] = []
    deductions: list[str] = []
    unsupported: list[str] = []
    for item, _ in kept:
        unsupported.extend(item.get("unsupported_citations", []))
        # A batch with no readable score sorts last rather than first.
        strength = -(item["parsed"]["ai_score"] or 0)
        for position, citation in enumerate(authorship_evidence(item["content"])):
            ranked.append((strength, position, citation))
        for deduction in deduction_claims(item["content"]):
            if deduction not in deductions:
                deductions.append(deduction)
    for _, _, citation in sorted(ranked, key=lambda entry: entry[:2]):
        if citation not in ai_citations:
            ai_citations.append(citation)

    # One summary, not one per batch. A human reads this report, and concatenating a batch
    # each repeated the whole narrative a dozen times over for a split repository.
    summary = ""
    for item, _ in kept:
        match = SUMMARY_PATTERN.search(item["content"])
        if match and match.group(1).strip():
            summary = match.group(1).strip()
            break
    deduction_text = "; ".join(deductions[:8]) or "no deductions applied"
    content = (
        f"1. AI Involvement Score (0-100%): {ai_score}%\n"
        f"Ranked AI evidence (strongest to weakest):\n"
        + "\n".join(
            f"Rank {index}: {citation}"
            for index, citation in enumerate(ai_citations[:8], 1)
        )
        + ("\nNo AI-specific evidence found." if not ai_citations else "")
        + "\n\n"
        f"2. Code Quality Score (0-100%): {code_score}%\n"
        f"- Deductions: {deduction_text}\n\n"
        f"3. Summary & Notable Findings: {summary}"
    )
    return {
        "status": 200,
        "content": content,
        "unsupported_citations": unsupported,
        "batch_count": len(reports),
        "batch_reports": [item["content"] for item in reports],
        "failed_batches": failed,
        "batch_scores": [
            {
                "ai_score": item["parsed"]["ai_score"],
                "code_score": item["parsed"]["code_score"],
                "source_lines": max(weight, 1),
            }
            for item, weight in zip(reports, source_line_weights, strict=True)
        ],
    }
