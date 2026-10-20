"""Measure model-visible features and assign evidence-weighted score targets.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import math
import statistics
from typing import Any

import detection as detector

from .config import (
    AI_LABEL_BAND,
    AI_LIKE_FEATURES,
    ARTIFACT_BAND,
    DOCUMENTATION_FEATURES,
    HUMAN_LABEL_BAND,
    HUMAN_LIKE_FEATURES,
    MAX_DOCUMENTATION_WEIGHT_SHARE,
    MIN_EFFECT,
    MIN_EVIDENCE_MARGIN,
)

__all__ = [
    "effect_sizes", "evidence_position", "evidence_score", "evidence_supports_label",
    "feature_supported", "feature_weights", "has_explicit_artifact",
    "label_target", "machine_likeness", "visible_features",
]

# A feature can only be read once the evidence it is measured over exists. A bare
# twenty-line snippet has no README, no docstrings and no comments, so those signals are
# structurally absent. Reading an absent site as perfect machine-likeness is what put
# mid-band targets on examples whose visible measurements carried no such evidence.
_SUPPORT_RULES: dict[str, tuple[str, int]] = {
    "readme structure": ("readme_chars", 400),
    "readme first person": ("readme_chars", 400),
    "readme informal": ("readme_chars", 400),
    "documented functions": ("functions", 2),
    "restated docstrings": ("docstrings", 1),
    "comment density": ("code_lines", 10),
    "human traces": ("source_lines", 20),
}


# Formatting is deliberately not a feature. Indentation and trailing whitespace were
# human-like features, which machine_likeness inverts, so a project indented with spaces
# scored as maximally machine-like for its zero tab count and the recorded build log showed
# that reading carrying 21% of a repository target. The prompt states that indentation is
# not authorship evidence, and the served model still answered "0 line(s) are tab indented"
# for ordinary projects, so the measurement is gone from both sides of the contract.
def visible_features(metrics: dict[str, Any]) -> dict[str, float]:
    """Project detector measurements into feature values visible to the model."""
    lines = max(1, metrics.get("code_lines") or 1)
    functions = metrics.get("functions") or 0
    docstrings = metrics.get("docstrings") or 0
    return {
        "readme structure": float((metrics.get("readme_headings") or 0) + (metrics.get("readme_bullets") or 0)),
        "readme first person": float(metrics.get("readme_first_person") or 0),
        "readme informal": float((metrics.get("readme_casual") or 0) + (metrics.get("readme_typos") or 0)),
        "documented functions": docstrings / functions if functions else 0.0,
        "restated docstrings": (metrics.get("restated_docstrings") or 0) / docstrings if docstrings else 0.0,
        "comment density": (metrics.get("comment_lines") or 0) / lines,
        "human traces": float(sum(metrics.get(name) or 0 for name in detector.TRACES)),
    }


def _gap(ai_values: list[float], human_values: list[float]) -> float:
    """Measure a pooled-standard-deviation difference between two feature groups."""
    spread = math.sqrt((statistics.pvariance(ai_values) + statistics.pvariance(human_values)) / 2)
    return (statistics.mean(ai_values) - statistics.mean(human_values)) / spread if spread else 0.0


def _clamp(value: float) -> float:
    """Clamp a normalized measurement to the unit interval."""
    return min(max(value, 0.0), 1.0)


def machine_likeness(metrics: dict[str, Any]) -> dict[str, float]:
    """Normalize visible features as evidence of machine authorship.

    A human-like feature contributes only what it observed. Inverting it outright read the
    absence of a first-person README, a TODO, or a debug print as maximal machine-likeness,
    which is the inference the prompt forbids and which the served judge reproduced: on
    ordinary repositories that one reading carried most of a target and returned high
    numbers for projects whose only distinguishing feature was being tidy. Observing none of
    a human trace is undecided, so it sits at the midpoint, and a trace that was observed
    still reads as fully human.
    """
    features = visible_features(metrics)
    values = {name: _clamp(features[name] / scale) for name, scale in AI_LIKE_FEATURES.items()}
    values.update(
        {
            name: 0.5 * (1.0 - _clamp(features[name] / scale))
            for name, scale in HUMAN_LIKE_FEATURES.items()
        }
    )
    return values


def effect_sizes(ai_rows: list[dict], human_rows: list[dict]) -> dict[str, float]:
    """Return absolute pooled-standard-deviation gaps for each visible feature."""
    return {
        name: abs(_gap([visible_features(row)[name] for row in ai_rows], [visible_features(row)[name] for row in human_rows]))
        for name in AI_LIKE_FEATURES | HUMAN_LIKE_FEATURES
    }


def _target_shares(usable: dict[str, float]) -> dict[str, float]:
    """Turn usable effect sizes into target weights, capped for documentation features.

    The measured separation decides a feature's share, except that the documentation
    features together may hold at most ``MAX_DOCUMENTATION_WEIGHT_SHARE`` of a target. A
    README is the project's own description of itself and costs nothing to write, so a
    target that mostly restates README shape teaches the grader to score the description
    rather than the code. When documentation is the only thing that separates the labels at
    all, the cap cannot be met by code features that do not exist and the measured share
    stands; ``feature_weights`` reports that case as an unsupported prompt shape.
    """
    total = sum(usable.values())
    if not total:
        return {}
    shares = {name: size / total for name, size in usable.items()}
    documentation = sum(
        share for name, share in shares.items() if name in DOCUMENTATION_FEATURES
    )
    if documentation <= MAX_DOCUMENTATION_WEIGHT_SHARE:
        return shares
    code_share = 1.0 - documentation
    if code_share <= 0.0:
        return shares
    documentation_scale = MAX_DOCUMENTATION_WEIGHT_SHARE / documentation
    code_scale = (1.0 - MAX_DOCUMENTATION_WEIGHT_SHARE) / code_share
    return {
        name: share
        * (documentation_scale if name in DOCUMENTATION_FEATURES else code_scale)
        for name, share in shares.items()
    }


def feature_weights(drafts: list[dict]) -> dict[str, dict[str, float]]:
    """Weight features with enough measured separation for each prompt shape."""
    weights = {}
    for kind in ("snippet", "repository"):
        groups = {label: [] for label in (0, 1)}
        for draft in drafts:
            if draft["label"] is not None and draft["kind"] == kind:
                groups[draft["label"]].append(draft["analysis"]["metrics"])
        effects = effect_sizes(groups[1], groups[0]) if all(groups.values()) else {}
        usable = {name: size for name, size in effects.items() if size >= MIN_EFFECT}
        weights[kind] = _target_shares(usable)
        if not weights[kind]:
            print(f"[!] {kind} rows: no feature separates the two labels by {MIN_EFFECT} standard deviations; different material is needed.")
    return weights


def feature_supported(metrics: dict[str, Any], name: str) -> bool:
    """Return whether the evidence a feature is measured over exists in this example."""
    rule = _SUPPORT_RULES.get(name)
    if rule is None:
        return True
    field, minimum = rule
    value = metrics.get(field)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= minimum


def evidence_position(metrics: dict[str, Any], weights: dict[str, float]) -> float:
    """Position an example within its score band using only readable measurements."""
    if not weights:
        return 0.5
    readable = {
        name: weight
        for name, weight in weights.items()
        if feature_supported(metrics, name)
    }
    if not readable:
        return 0.5
    values = machine_likeness(metrics)
    total = sum(readable.values())
    return _clamp(
        sum(weight * values[name] for name, weight in readable.items()) / total
    )


def evidence_supports_label(
    position: float, label: int, *, margin: float = MIN_EVIDENCE_MARGIN
) -> bool:
    """Return whether a visible position sits on the label's side of the midpoint."""
    if label:
        return position >= 0.5 + margin
    return position <= 0.5 - margin


def has_explicit_artifact(analysis: dict[str, Any]) -> bool:
    """Return whether the detector found an explicit AI-generation artifact."""
    return any((finding.get("weight") or 0) >= detector.EXPLICIT_ARTIFACT_WEIGHT for finding in analysis.get("ai_findings") or [])


def evidence_score(analysis: dict[str, Any], weights: dict[str, float]) -> int:
    """Map any example's own measurements to the one score target every row shares.

    Labeled and unlabeled examples go through this single function, so two rows carrying
    the same measurements cannot land in different score regimes: the prompt's visible
    position picks the band and the distance from an undecided reading spreads the example
    across that band. An explicit AI-generation marker outranks the position, because no
    reading of the other measurements outweighs a line of the source saying it was
    generated.

    Distance, not position, carries the target. Mapping the position onto the band directly
    wasted most of it: a supervised AI example had to reach 0.6 and the reachable range
    therefore began at 69, so nearly every repository row targeted the same high score and
    the served grader returned one number for every project. A position at the midpoint now
    sits on the band edge nearest the decision boundary and an extreme position reaches the
    far edge, so a weak AI reading targets 53 and a strong one 85.
    """
    position = evidence_position(analysis["metrics"], weights)
    intensity = abs(position - 0.5) * 2
    if has_explicit_artifact(analysis):
        low, high = ARTIFACT_BAND
        return round(low + intensity * (high - low))
    if position >= 0.5:
        low, high = AI_LABEL_BAND
        return round(low + intensity * (high - low))
    low, high = HUMAN_LABEL_BAND
    return high - round(intensity * (high - low))


def label_target(
    analysis: dict[str, Any], label: int, weights: dict[str, float]
) -> int | None:
    """Return the shared target for a labeled example, or ``None`` when it disagrees.

    A dataset label the prompt's own measurements contradict cannot be learned from the
    prompt: the model is shown evidence that points the other way and is asked to output
    the opposite band anyway. Returning ``None`` lets the builder drop those rows instead
    of teaching the model to disregard the measurements it is shown. Every surviving
    label takes its score from ``evidence_score``, exactly like an unlabeled example.
    """
    if not has_explicit_artifact(analysis) and not evidence_supports_label(
        evidence_position(analysis["metrics"], weights), label
    ):
        return None
    return evidence_score(analysis, weights)
