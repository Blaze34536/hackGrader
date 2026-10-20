"""Render evidence examples into the shared live grading conversation format.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from typing import Any

import contracts as report
import detection as detector
from contracts.prompts import FILE_NAME_HEADER, MODEL_DUMP_CHARS

from .config import FLAG_THRESHOLD
from .validation import DEFAULT_MAX_LEN


def core(
    prompt_code: str,
    url: str,
    *,
    seed: int = 0,
    analysis: dict[str, Any] | None = None,
    label: int | None = None,
    verified_label: str | None = None,
    verified_entry: dict[str, Any] | None = None,
    kind: str = "snippet",
    source_type: str = "synthetic_repository",
    source_groups: list[str] | None = None,
) -> dict[str, Any]:
    """Retain evidence, target label, and source lineage for later rendering.

    ``label`` is a style-derived class the example must itself support. A
    ``verified_label`` is an audited authorship record, which is set instead when the
    repository's provenance, not its prompt text, decides the target; ``verified_entry``
    carries that record's own measurement so the target is the share it attributed rather
    than a band constant.
    """
    return {
        "prompt": prompt_code,
        "url": url,
        "seed": seed,
        "label": label,
        "verified_label": verified_label,
        "verified_entry": verified_entry,
        "kind": kind,
        "source_type": source_type,
        "source_groups": source_groups or [url],
        "analysis": analysis or detector.analyze(prompt_code),
    }


def _token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    """Count tokens using the exact chat serialization used by training preflight."""
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def batch_measurement(entry: dict[str, Any] | None, batch: str) -> dict[str, int] | None:
    """Return the attributed and examined line counts of the files one batch actually shows.

    A repository's share is taken over every file it examined, but a view shows only some of
    them. Taking the share over the shown files makes each batch's target rest on its own code,
    so one repository gives many targets rather than one. ``None`` means the manifest records no
    per-file counts for these files, and the caller keeps the repository share.
    """
    files = entry.get("files") if entry else None
    if not files:
        return None
    shown = {name.strip() for name in FILE_NAME_HEADER.findall(batch)}
    attributed = examined = 0
    for item in files:
        if item["path"] in shown:
            attributed += int(item["ai_lines"])
            examined += int(item["examined_lines"])
    if examined <= 0:
        return None
    return {"attributed_lines": attributed, "examined_lines": examined}


def _finalize_core(
    item: dict[str, Any],
    weights: dict[str, dict[str, float]],
    *,
    budget: int | None,
    unsupported: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Render one source into report rows at the requested evidence character budget."""
    from .evidence import VERIFIED_EVIDENCE_WEIGHTS, label_evidence
    from .features import evidence_score, label_target
    from .labels import LABEL_AI, provenance_target

    batches = report.split_dump(item["prompt"], budget=budget)
    if not batches:
        return []
    question_base = report.BASE_QUESTION.format(url=item["url"])
    rows = []
    for index, batch in enumerate(batches):
        analysis = detector.analyze(batch)
        label = item["label"]
        row_weights = weights.get(item["kind"], {})
        verified = item.get("verified_label")
        if verified is not None:
            entry = item.get("verified_entry")
            target = provenance_target(
                verified, analysis, batch_measurement(entry, batch) or entry
            )
            if target is None:
                if unsupported is not None:
                    key = str(item["source_type"])
                    unsupported[key] = unsupported.get(key, 0) + 1
                continue
            evidence = label_evidence(
                batch, int(verified == LABEL_AI), analysis, VERIFIED_EVIDENCE_WEIGHTS
            )
        elif label is None:
            target = evidence_score(analysis, row_weights)
            evidence = label_evidence(
                batch,
                int(target >= FLAG_THRESHOLD),
                analysis,
                row_weights,
            )
        else:
            target = label_target(analysis, label, row_weights)
            if target is None:
                if unsupported is not None:
                    key = str(item["source_type"])
                    unsupported[key] = unsupported.get(key, 0) + 1
                continue
            evidence = label_evidence(batch, label, analysis, row_weights)
        if not evidence:
            # The view offers nothing to quote, so any target for it would have to invent a
            # quote. Drop the batch and report it like any other unsupported label.
            if unsupported is not None:
                key = str(item["source_type"])
                unsupported[key] = unsupported.get(key, 0) + 1
            continue
        question = report.batch_question(question_base, index + 1, len(batches))
        answer = detector.build_report(
            batch,
            seed=item["seed"] + index,
            ai_score=target,
            ai_evidence=evidence,
        )
        rows.append({
            "messages": [
                {"role": "system", "content": report.SYSTEM_PROMPT},
                {"role": "user", "content": report.build_user_message(
                    question, batch, url=item["url"], analysis=analysis,
                )},
                {"role": "assistant", "content": answer},
            ],
            "source_url": item["url"],
            "source_type": item["source_type"],
            "source_kind": item["kind"],
            "source_groups": list(item["source_groups"]),
        })
    return rows


def finalize(
    cores: list[dict[str, Any]],
    weights: dict[str, dict[str, float]],
    *,
    tokenizer: Any | None = None,
    max_tokens: int = DEFAULT_MAX_LEN,
) -> list[dict[str, Any]]:
    """Build source-aware rows and adapt batch size to the configured token window."""
    rows = []
    unsupported: dict[str, int] = {}
    for item in cores:
        if tokenizer is None:
            rows.extend(
                _finalize_core(item, weights, budget=None, unsupported=unsupported)
            )
            continue

        budget = MODEL_DUMP_CHARS
        while True:
            core_rows = _finalize_core(
                item, weights, budget=budget, unsupported=unsupported
            )
            lengths = [
                _token_count(tokenizer, row["messages"]) for row in core_rows
            ]
            longest = max(lengths, default=0)
            if longest <= max_tokens:
                rows.extend(core_rows)
                if budget < MODEL_DUMP_CHARS:
                    print(
                        f"[i] Reduced {item['source_type']} evidence batches to "
                        f"{budget} characters to fit {max_tokens} tokens "
                        f"({len(core_rows)} batch(es), longest {longest} tokens)."
                    )
                break

            next_budget = int(budget * max_tokens / longest * 0.9)
            if next_budget >= budget:
                next_budget = budget - 1
            if next_budget < 256:
                raise ValueError(
                    f"Cannot fit {item['source_type']} source {item['url']} within "
                    f"{max_tokens} tokens without dropping repository headers."
                )
            budget = next_budget
    if unsupported:
        dropped = sum(unsupported.values())
        detail = ", ".join(
            f"{name}={count}" for name, count in sorted(unsupported.items())
        )
        print(
            f"[i] skipped {dropped} labeled evidence batch(es) whose own measurements do "
            f"not support their dataset label ({detail}); supervising them would teach "
            "the model to disregard the evidence it is shown."
        )
    return rows


def usable(row: dict[str, Any]) -> bool:
    """Return whether the generated assistant target is valid and cited."""
    parsed = report.parse_report(row["messages"][-1]["content"])
    return parsed["valid"] and bool(parsed["citations"])
