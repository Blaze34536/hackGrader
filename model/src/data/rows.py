"""Summarize, filter, and deduplicate finalized candidate rows.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import collections
import hashlib
import math
import random
import statistics
from typing import Any

import contracts as report
import detection as detector

from .config import FLAG_THRESHOLD, MAX_TARGET_SHARE
from .features import visible_features
from .splitting import _components
from .validation import canonical_source_url


def _feature_gap(ai_values: list[float], human_values: list[float]) -> float:
    """Measure the mean difference in units of pooled standard deviation."""
    pooled = math.sqrt((statistics.pvariance(ai_values) + statistics.pvariance(human_values)) / 2)
    return (statistics.mean(ai_values) - statistics.mean(human_values)) / pooled if pooled else 0.0


def score_of(row: dict[str, Any]) -> int:
    """Return a parseable assistant AI target without inventing a default label."""
    score = report.parse_report(row["messages"][-1]["content"])["ai_score"]
    if score is None:
        raise ValueError("candidate row has no parseable assistant AI target")
    return score


def is_snippet(row: dict[str, Any]) -> bool:
    """Identify AIGCodeSet evidence examples by their source metadata."""
    return row.get("source_type") == "aigcodeset" or "aigcodeset" in row["messages"][1]["content"][:400].lower()


def is_machine_written(row: dict[str, Any]) -> bool:
    """Return whether a row's target score sits on the machine-written side of the flag."""
    return score_of(row) >= FLAG_THRESHOLD


def cap_targets(rows: list[dict[str, Any]], cap: int, rng: random.Random) -> list[dict[str, Any]]:
    """Limit each target score bucket to avoid a single score dominating the pool."""
    buckets: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        buckets[score_of(row)].append(row)
    kept = []
    for score in sorted(buckets):
        bucket = buckets[score]
        rng.shuffle(bucket)
        kept.extend(bucket[:cap] if cap > 0 else bucket)
    rng.shuffle(kept)
    return kept


def quarantine_conflicting_prompts(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Exclude contradictory prompts and every row linked to their source components."""
    targets_by_prompt: dict[str, set[int]] = collections.defaultdict(set)
    rows_by_prompt: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        prompt = row["messages"][1]["content"]
        targets_by_prompt[prompt].add(score_of(row))
        rows_by_prompt[prompt].append(index)

    conflicts = {
        prompt: targets
        for prompt, targets in targets_by_prompt.items()
        if len(targets) > 1
    }
    if not conflicts:
        return rows, 0

    conflicting_components: set[int] = set()
    quarantined: set[int] = set()
    for component_index, component in enumerate(_components(rows)):
        if any(
            rows[index]["messages"][1]["content"] in conflicts
            for index in component
        ):
            conflicting_components.add(component_index)
            quarantined.update(component)

    details = []
    for prompt, scores in sorted(conflicts.items(), key=lambda item: item[0]):
        conflict_rows = [rows[index] for index in rows_by_prompt[prompt]]
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        sources = sorted({str(row.get("source_type", "unknown")) for row in conflict_rows})
        source_urls = sorted(
            {
                canonical_source_url(str(row.get("source_url", row.get("url", ""))))
                for row in conflict_rows
            }
        )
        shown_urls = source_urls[:3]
        if len(source_urls) > len(shown_urls):
            shown_urls.append(f"+{len(source_urls) - len(shown_urls)} more")
        details.append(
            f"{digest}: scores={','.join(map(str, sorted(scores)))} "
            f"sources={','.join(sources)} urls={','.join(shown_urls)}"
        )
    details_text = "; ".join(details[:8])
    if len(details) > 8:
        details_text += f"; +{len(details) - 8} more conflicting prompts"
    print(
        f"[!] Quarantined {len(conflicts)} prompt(s) with conflicting AI targets and "
        f"{len(quarantined)} row(s) in {len(conflicting_components)} connected "
        f"source component(s): {details_text}"
    )
    return [row for index, row in enumerate(rows) if index not in quarantined], len(quarantined)


def distinct(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep unique rendered prompts and reject contradictory targets if not quarantined."""
    seen: dict[str, tuple[int, dict[str, Any]]] = {}
    unique = []
    for row in rows:
        key = row["messages"][1]["content"]
        score = score_of(row)
        if key in seen:
            previous_score, previous_row = seen[key]
            if previous_score != score:
                sources = sorted(
                    {
                        f"{item.get('source_type', 'unknown')}@{canonical_source_url(str(item.get('source_url', item.get('url', ''))))}"
                        for item in (previous_row, row)
                    }
                )
                raise ValueError(
                    f"duplicate prompt has conflicting AI targets ({previous_score} vs {score}) "
                    f"from {sources}; run global source-linked prompt quarantine before "
                    "source-kind deduplication"
                )
            merged_groups = sorted(
                set(previous_row.get("source_groups", []))
                | set(row.get("source_groups", []))
            )
            previous_row["source_groups"] = merged_groups
            continue
        kept = dict(row)
        kept["source_groups"] = list(row.get("source_groups", []))
        seen[key] = (score, kept)
        unique.append(kept)
    return unique, len(rows) - len(unique)


def label_histogram(rows: list[dict[str, Any]]) -> dict[int, int]:
    """Count target scores in ascending order for candidate diagnostics."""
    return dict(sorted(collections.Counter(score_of(row) for row in rows).items()))


def report_mix(rows: list[dict[str, Any]], repository_rows: int, spread: dict[int, int]) -> None:
    """Summarize prompt shape, label balance, and the dominant score bucket."""
    if not rows:
        print("[!] The dataset has no rows after filtering.")
        return
    snippet_rows = len(rows) - repository_rows
    print(f"[i] mix: {snippet_rows} snippet row(s) and {repository_rows} repository-format row(s) ({repository_rows / len(rows):.0%} runtime-shaped)")
    machine = sum(is_machine_written(row) for row in rows)
    print(
        f"[i] AI score targets: {spread} ({machine} machine-written, "
        f"{len(rows) - machine} hand-written)"
    )
    if not spread:
        return
    common, count = max(spread.items(), key=lambda item: item[1])
    if count / len(rows) > MAX_TARGET_SHARE:
        print(f"[!] {count / len(rows):.0%} of rows target {common}%; add more varied examples.")
    if max(spread) < 60:
        print(f"[!] Highest target is {max(spread)}%; the AI high band is missing.")
    below = sum(count for score, count in spread.items() if score < FLAG_THRESHOLD)
    above = sum(count for score, count in spread.items() if score >= FLAG_THRESHOLD)
    if not below or not above:
        print("[!] Score targets do not cover both score classes.")


def report_separation(rows: list[dict[str, Any]], weights: dict[str, dict[str, float]], limit: int = 100000) -> None:
    """Compare visible features across labeled classes for both prompt shapes."""
    for kind in ("snippet", "repository"):
        groups = {label: [] for label in (0, 1)}
        for row in rows:
            if is_snippet(row) == (kind == "snippet"):
                label = int(is_machine_written(row))
                if len(groups[label]) < limit:
                    groups[label].append(row)
        if not groups[1] or not groups[0]:
            continue
        ai_metrics = [detector.analyze(prompt_block(row))["metrics"] for row in groups[1]]
        human_metrics = [detector.analyze(prompt_block(row))["metrics"] for row in groups[0]]
        print(f"[i] {kind} labels: AI {sum(map(score_of, groups[1])) / len(groups[1]):.0f}% vs human {sum(map(score_of, groups[0])) / len(groups[0]):.0f}%")
        for name in visible_features(ai_metrics[0]):
            gap = _feature_gap([visible_features(m)[name] for m in ai_metrics], [visible_features(m)[name] for m in human_metrics])
            share = weights.get(kind, {}).get(name)
            note = f"carries {share:.0%} of target" if share else "not weighted"
            print(f"      {name:20s} d={gap:+.2f} {note}")


def prompt_block(row: dict[str, Any]) -> str:
    """Extract only the delimited evidence block the model actually sees."""
    user = row["messages"][1]["content"]
    begin = "===== BEGIN REPOSITORY CONTENT =====\n"
    end = "\n===== END REPOSITORY CONTENT ====="
    if begin not in user or end not in user:
        return ""
    return user.split(begin, 1)[1].split(end, 1)[0]
