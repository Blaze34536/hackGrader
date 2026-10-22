"""Reserve whole lineage components until every partition meets its quotas.

Components are indivisible because splitting one would leak the same source material
across partitions, so the allocator may only skip a component, never break it up. It
therefore has to look ahead: a component is only placed somewhere if enough supply is
left to satisfy every quota still outstanding, everywhere.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from .bands import _CLASS_LABELS, _SCORE_BANDS, _SCORE_CLASSES
from .capacity import CapacityReport, category_index
from .components import ComponentProfile
from .priority import component_priority
from .progress import PartitionProgress, Requirement
from .quotas import PartitionQuotas


class ComponentAllocator:
    """Hand out complete lineage components, one outstanding quota at a time."""

    def __init__(
        self,
        profiles: list[ComponentProfile],
        quotas: PartitionQuotas,
        capacity: CapacityReport,
        holdout_fractions: tuple[tuple[str, float], ...],
    ) -> None:
        """Seed the running counters every assignment updates."""
        self.profiles = profiles
        self.quotas = quotas
        self.holdout_fractions = holdout_fractions
        self.partition_names = quotas.partition_names
        self.required_categories = capacity.required_categories
        self.required_category_partitions = capacity.required_category_partitions
        self.category_components = category_index(profiles, self.required_categories)
        self.assigned: dict[int, str] = {}
        self.progress = {
            name: PartitionProgress(name) for name in self.partition_names
        }

        self.remaining_prompt_supply = sum(
            profile.prompt_count for profile in profiles
        )
        self.remaining_class_supply = {
            label: sum(profile.classes[label] for profile in profiles)
            for label in _CLASS_LABELS
        }
        self.remaining_band_supply = {
            band: sum(profile.bands[band] for profile in profiles)
            for band in _SCORE_BANDS
        }
        self.remaining_repository_supply = capacity.real_repository_components
        self.remaining_real_repository_prompt_supply = capacity.real_repository_prompts
        self.remaining_real_repository_class_supply = {
            label: sum(profile.real_repository_classes[label] for profile in profiles)
            for label in _SCORE_CLASSES
        }
        self.remaining_category_supply = {
            category: len(indices)
            for category, indices in self.category_components.items()
        }

    def contributes(self, index: int, requirement: Requirement) -> bool:
        """Whether one component can reduce an outstanding quota."""
        kind, label, _ = requirement
        profile = self.profiles[index]
        if kind == "source":
            return label in profile.types
        if kind == "class":
            return profile.classes[label] > 0
        if kind == "repository_class":
            return profile.real_repository_classes[label] > 0
        if kind == "repository_prompt":
            return bool(profile.real_repository_prompts)
        if kind == "band":
            return profile.bands[label] > 0
        if kind == "repository":
            return profile.has_repository
        return bool(profile.prompts)

    def can_assign(self, index: int, destination: str) -> bool:
        """Preserve enough unassigned components and examples for every remaining quota."""
        profile = self.profiles[index]
        for category in self.required_categories:
            still_needing = sum(
                name in self.required_category_partitions[category]
                and category not in self.progress[name].covered
                and not (name == destination and category in profile.categories)
                for name in self.partition_names
            )
            free_components = self.remaining_category_supply[category] - int(
                category in profile.categories
            )
            if free_components < still_needing:
                return False

        prompts_needed = sum(
            max(
                0,
                self.quotas.requested_rows[name]
                - self.progress[name].prompts
                - (profile.prompt_count if name == destination else 0),
            )
            for name in self.partition_names
        )
        if self.remaining_prompt_supply - profile.prompt_count < prompts_needed:
            return False

        for label in _CLASS_LABELS:
            supply = self.remaining_class_supply[label] - profile.classes[label]
            needed = sum(
                max(
                    0,
                    self.quotas.class_minimums[name][label]
                    - self.progress[name].classes[label]
                    - (profile.classes[label] if name == destination else 0),
                )
                for name in self.partition_names
            )
            if supply < needed:
                return False

        for band in _SCORE_BANDS:
            supply = self.remaining_band_supply[band] - profile.bands[band]
            needed = sum(
                max(
                    0,
                    self.quotas.band_minimums[name][band]
                    - self.progress[name].bands[band]
                    - (profile.bands[band] if name == destination else 0),
                )
                for name in self.partition_names
            )
            if supply < needed:
                return False

        repository_supply = self.remaining_repository_supply - int(
            profile.has_repository
        )
        repository_needed = sum(
            max(
                0,
                self.quotas.real_repository_minimums[name]
                - self.progress[name].repositories
                - (int(profile.has_repository) if name == destination else 0),
            )
            for name in self.partition_names
        )
        if repository_supply < repository_needed:
            return False

        prompt_supply = (
            self.remaining_real_repository_prompt_supply
            - len(profile.real_repository_prompts)
        )
        prompt_needed = sum(
            max(
                0,
                self.quotas.real_repository_prompt_minimums[name]
                - len(self.progress[name].real_repository_prompts)
                - (
                    len(profile.real_repository_prompts)
                    if name == destination
                    else 0
                ),
            )
            for name in self.partition_names
        )
        if prompt_supply < prompt_needed:
            return False

        for label in _SCORE_CLASSES:
            repository_supply = (
                self.remaining_real_repository_class_supply[label]
                - profile.real_repository_classes[label]
            )
            repository_needed = sum(
                max(
                    0,
                    self.quotas.real_repository_class_minimums[name][label]
                    - self.progress[name].real_repository_classes[label]
                    - (
                        profile.real_repository_classes[label]
                        if name == destination
                        else 0
                    ),
                )
                for name in self.partition_names
            )
            if repository_supply < repository_needed:
                return False

        return True

    def assign(self, index: int, destination: str) -> None:
        """Reserve a complete lineage component for one partition."""
        profile = self.profiles[index]
        self.assigned[index] = destination
        self.progress[destination].record(profile)
        self.remaining_prompt_supply -= profile.prompt_count
        self.remaining_repository_supply -= int(profile.has_repository)
        self.remaining_real_repository_prompt_supply -= len(profile.real_repository_prompts)
        for label, count in profile.real_repository_classes.items():
            if label in self.remaining_real_repository_class_supply:
                self.remaining_real_repository_class_supply[label] -= count
        for category in profile.categories & self.required_categories:
            self.remaining_category_supply[category] -= 1
        for label, count in profile.classes.items():
            self.remaining_class_supply[label] -= count
        for band, count in profile.bands.items():
            self.remaining_band_supply[band] -= count

    def supply_for(self, requirement: Requirement) -> int:
        """Count unassigned components that could satisfy one quota at all."""
        return sum(
            index not in self.assigned and self.contributes(index, requirement)
            for index in range(len(self.profiles))
        )

    def candidates_for(self, requirement: Requirement, destination: str) -> list[int]:
        """Unassigned components that can fill one quota and keep the rest feasible."""
        return [
            index
            for index in range(len(self.profiles))
            if index not in self.assigned
            and self.contributes(index, requirement)
            and self.can_assign(index, destination)
        ]

    def choose_requirement(
        self, missing: list[Requirement], destination: str
    ) -> Requirement:
        """Pick the scarcest quota that still has a feasible component to fill it."""
        order = sorted(
            missing,
            key=lambda requirement: (
                self.supply_for(requirement),
                requirement[0],
                requirement[1],
            ),
        )
        for requirement in order:
            if self.candidates_for(requirement, destination):
                return requirement
        return order[-1]

    def stratify(self, destination: str, fraction: float) -> None:
        """Give one holdout its proportional share of every stratum before any quota.

        The quota loop alone picks the component that pays off the most deficits, and a
        repository whose views span both classes pays off the most, so holdouts filled that
        way were 75-80% mostly-AI repositories against 34% in training. Drawing each
        stratum in proportion first keeps every partition the same mix; the quota loop then
        only tops up what is still missing.
        """
        strata: dict[tuple[str, str], list[int]] = {}
        for index, profile in enumerate(self.profiles):
            strata.setdefault(profile.stratum, []).append(index)
        for members in strata.values():
            # Prompts, not components: holdouts are sized in prompts, and counting components
            # let the quota top-up pull in the small, mostly-AI repositories to make up rows.
            wanted = fraction * sum(self.profiles[index].prompt_count for index in members)
            for index in members:
                if wanted < self.profiles[index].prompt_count / 2:
                    continue
                if index not in self.assigned and self.can_assign(index, destination):
                    self.assign(index, destination)
                    wanted -= self.profiles[index].prompt_count

    def run(self) -> dict[int, str]:
        """Fill every holdout, leaving whatever is unassigned for training."""
        for destination, fraction in self.holdout_fractions:
            self.stratify(destination, fraction)
            while missing := self.progress[destination].deficits(self.quotas):
                requirement = self.choose_requirement(missing, destination)
                candidates = self.candidates_for(requirement, destination)
                kind, label, _ = requirement
                if not candidates:
                    raise ValueError(
                        f"could not reserve a source-disjoint {kind}={label!r} component "
                        f"for {destination}"
                    )
                outstanding = tuple(self.progress[destination].deficits(self.quotas))
                self.assign(
                    max(
                        candidates,
                        key=lambda index: component_priority(
                            self.profiles[index], index, outstanding
                        ),
                    ),
                    destination,
                )
        return self.assigned
