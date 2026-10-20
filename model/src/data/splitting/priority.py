"""Score how good a given component is at closing the deficits a partition still has.

Placing a whole component is a one-way decision, so the allocator prefers components
that pay off several open quotas at once and are cheap to spend. Spanning many prompts
does not help either partition, so smaller components win ties.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from .components import ComponentProfile
from .progress import Requirement


def _contribution(profile: ComponentProfile, requirement: Requirement) -> int:
    """Return how many examples one component offers toward a single quota."""
    kind, label, _ = requirement
    if kind == "source":
        return int(label in profile.types)
    if kind == "class":
        return profile.classes[label]
    if kind == "band":
        return profile.bands[label]
    if kind == "repository":
        return len(profile.repositories)
    if kind == "repository_prompt":
        return len(profile.real_repository_prompts)
    if kind == "repository_class":
        return profile.real_repository_classes[label]
    return profile.prompt_count


def component_priority(
    profile: ComponentProfile, index: int, outstanding: tuple[Requirement, ...]
) -> tuple[float, int, int]:
    """Prefer useful, compact components that reduce several current deficits."""
    benefit = 0.0
    for requirement in outstanding:
        _, _, count = requirement
        contribution = _contribution(profile, requirement)
        benefit += min(contribution, count) / max(count, 1)
    return (benefit, -profile.prompt_count, -index)
