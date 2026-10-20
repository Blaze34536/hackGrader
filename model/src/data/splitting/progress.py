"""Track what one partition has accumulated and which of its floors are still unmet.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..validation import REQUIRED_SPLIT_SOURCE_TYPES
from .bands import _CLASS_LABELS, _SCORE_BANDS, _SCORE_CLASSES
from .components import ComponentProfile
from .quotas import PartitionQuotas

Requirement = tuple[str, str, int]


@dataclass
class PartitionProgress:
    """Coverage and example counts for a single partition as it is filled."""

    name: str
    covered: set[str] = field(default_factory=set)
    prompts: int = 0
    repositories: int = 0
    classes: dict[str, int] = field(
        default_factory=lambda: {label: 0 for label in _CLASS_LABELS}
    )
    real_repository_prompts: set[str] = field(default_factory=set)
    real_repository_classes: dict[str, int] = field(
        default_factory=lambda: {label: 0 for label in _CLASS_LABELS}
    )
    bands: dict[str, int] = field(
        default_factory=lambda: {band: 0 for band in _SCORE_BANDS}
    )

    def record(self, profile: ComponentProfile) -> None:
        """Add one reserved component's contribution to this partition."""
        self.covered.update(profile.categories)
        self.prompts += profile.prompt_count
        self.repositories += len(profile.repositories)
        self.real_repository_prompts.update(profile.real_repository_prompts)
        for label, count in profile.classes.items():
            self.classes[label] += count
        for label, count in profile.real_repository_classes.items():
            self.real_repository_classes[label] += count
        for band, count in profile.bands.items():
            self.bands[band] += count

    def deficits(self, quotas: PartitionQuotas) -> list[Requirement]:
        """Return every coverage and numeric quota this partition still misses."""
        requirements: list[Requirement] = [
            ("source", source_type, 1)
            for source_type in sorted(REQUIRED_SPLIT_SOURCE_TYPES - self.covered)
        ]
        requirements.extend(
            ("class", label, quotas.class_minimums[self.name][label] - self.classes[label])
            for label in _CLASS_LABELS
            if self.classes[label] < quotas.class_minimums[self.name][label]
        )
        requirements.extend(
            ("band", band, quotas.band_minimums[self.name][band] - self.bands[band])
            for band in _SCORE_BANDS
            if self.bands[band] < quotas.band_minimums[self.name][band]
        )
        minimum = quotas.real_repository_minimums[self.name]
        if self.repositories < minimum:
            requirements.append(
                ("repository", "real_repository", minimum - self.repositories)
            )
        repository_prompts = quotas.real_repository_prompt_minimums[self.name]
        if len(self.real_repository_prompts) < repository_prompts:
            requirements.append(("repository_prompt", "real", repository_prompts - len(self.real_repository_prompts)))
        for label in _SCORE_CLASSES:
            minimum = quotas.real_repository_class_minimums[self.name][label]
            if self.real_repository_classes[label] < minimum:
                requirements.append(
                    (
                        "repository_class",
                        label,
                        minimum - self.real_repository_classes[label],
                    )
                )
        if self.prompts < quotas.requested_rows[self.name]:
            requirements.append(
                ("rows", "all", quotas.requested_rows[self.name] - self.prompts)
            )
        return requirements
