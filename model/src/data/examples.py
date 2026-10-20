"""Load and wrap labeled AIGCodeSet samples with stable source lineage.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import hashlib
import random
import zlib
from typing import Any, NamedTuple

import pandas as pd

from .config import AIG_CSV, AIG_ROWS, MIN_SNIPPET_CHARS, SNIPPET_CHARS
from .rendering import core


class AIGSample(NamedTuple):
    """One official labeled code sample and its CodeNet problem lineage."""

    label: int
    code: str
    problem_group: str
    problem_groups: tuple[str, ...] = ()


def aig_samples(limit: int = AIG_ROWS) -> list[AIGSample]:
    """Load balanced, bounded code samples from the labeled AIGCodeSet corpus."""
    if limit <= 0:
        return []
    try:
        frame = pd.read_csv(
            AIG_CSV,
            usecols=lambda name: name
            in {"code", "text", "label", "problem_id"},
        )
    except (OSError, ValueError) as exc:
        print(f"[!] AIGCodeSet unavailable ({exc}); continuing without it.")
        return []
    column = next((name for name in ("code", "text") if name in frame.columns), None)
    if column is None or "label" not in frame.columns or "problem_id" not in frame.columns:
        print(f"[!] AIGCodeSet columns are unexpected: {list(frame.columns)[:8]}")
        return []
    frame = frame.dropna(subset=[column, "problem_id"])
    samples: list[AIGSample] = []
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    observed_labels = set(frame["label"].dropna().unique())
    if not observed_labels <= {0, 1} or observed_labels != {0, 1}:
        raise SystemExit(
            "[!] AIGCodeSet labels are not the verified binary 0=human/1=LLM "
            f"mapping: found {sorted(observed_labels)}. Refusing to generate supervision."
        )
    candidates: dict[int, dict[str, set[str]]] = {1: {}, 0: {}}
    labels_by_code: dict[str, set[int]] = {}
    for code_value, label_value, problem_id in zip(
        frame[column], frame["label"], frame["problem_id"], strict=True
    ):
        if label_value not in (0, 1):
            continue
        label = int(label_value)
        code = str(code_value).strip()[:SNIPPET_CHARS]
        if len(code) < MIN_SNIPPET_CHARS:
            continue
        problem_text = str(problem_id).strip().casefold()
        if not problem_text:
            continue
        problem_group = hashlib.sha256(problem_text.encode("utf-8")).hexdigest()
        candidates[label].setdefault(code, set()).add(problem_group)
        labels_by_code.setdefault(code, set()).add(label)

    conflicting_codes = {
        code for code, labels in labels_by_code.items() if len(labels) > 1
    }
    conflicting_problem_groups = {
        problem_group
        for code in conflicting_codes
        for label in (0, 1)
        for problem_group in candidates[label].get(code, set())
    }
    if conflicting_codes:
        print(
            f"[!] AIGCodeSet: quarantined {len(conflicting_codes)} code prompt(s) "
            "that have both human and LLM labels after prompt truncation, plus "
            f"{len(conflicting_problem_groups)} linked problem group(s); no "
            "samples or synthetic derivatives from those tasks will be generated."
        )

    for label in (1, 0):
        codes = sorted(
            code
            for code, problem_groups in candidates[label].items()
            if code not in conflicting_codes
            and not problem_groups.intersection(conflicting_problem_groups)
        )
        selected = random.Random(42).sample(codes, min(limit, len(codes)))
        for code in selected:
            problem_groups = tuple(sorted(candidates[label][code]))
            samples.append(
                AIGSample(label, code, problem_groups[0], problem_groups)
            )
        meaning = "LLM-generated" if label == 1 else "human-written"
        print(
            f"[i] AIGCodeSet label {label} ({meaning}): "
            f"{len(selected)} unique candidate sample(s)"
        )
        if not selected:
            raise SystemExit(
                f"[!] AIGCodeSet label {label} ({meaning}) produced no usable samples."
            )
    return samples


def sample_source_groups(sample: AIGSample) -> list[str]:
    """Return content- and problem-level lineage keys for one AIGCodeSet sample."""
    identity = hashlib.sha256(sample.code.encode("utf-8")).hexdigest()
    problem_groups = sample.problem_groups or (sample.problem_group,)
    return [
        f"aigcodeset-row:{identity}",
        *(f"aigcodeset-problem:{group}" for group in problem_groups),
    ]


def snippet_rows(samples: list[AIGSample]) -> list[dict[str, Any]]:
    """Wrap labeled code samples in one-file views with stable source/task lineage."""
    rows = []
    for sample in samples:
        seed = zlib.crc32(sample.code.encode("utf-8"))
        identity = hashlib.sha256(sample.code.encode("utf-8")).hexdigest()
        rows.append(core(
            f"--- FILE: snippet.py ---\n{sample.code}",
            f"https://github.com/code-review-samples/sample-{identity[:16]}",
            seed=seed, label=sample.label, kind="snippet", source_type="aigcodeset",
            source_groups=sample_source_groups(sample),
        ))
    return rows
