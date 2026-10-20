"""Bound a harvest by the requests it may spend and record what it spent.

The cohort's size is set by the rate limit, not by ambition: GitHub answers 5,000 core
requests an hour for a token and 60 for an anonymous caller, and the per-file last-touch
rule costs one request per file examined. A harvest that ran until GitHub refused it would
stop in the middle of a repository with no record of why, so the ceiling is explicit, every
request is counted, and reaching the ceiling stops the run cleanly with the cohort and its
spent budget written out.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """Raised when a harvest has spent the requests it was allowed."""


@dataclass
class RequestBudget:
    """A request ceiling with the calls made against it and the limits observed."""

    ceiling: int = 5000
    spent: int = 0
    raw_spent: int = 0
    stopped: str = ""
    rate_limit: int = 0
    rate_remaining: int = 0
    rate_reset: int = 0
    rate_resource: str = ""
    ceiling_hits: int = 0
    _observations: list[dict[str, int | str]] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        """Return how many metered requests are still allowed."""
        return max(self.ceiling - self.spent, 0)

    def spend(self, count: int = 1) -> None:
        """Count metered requests, refusing the call that would pass the ceiling."""
        if self.spent + count > self.ceiling:
            self.ceiling_hits += 1
            self.stopped = (
                f"request ceiling reached ({self.spent} of {self.ceiling} spent); "
                "the cohort keeps every repository that finished"
            )
            raise BudgetExceeded(self.stopped)
        self.spent += count

    def note_raw(self, count: int = 1) -> None:
        """Count a content fetch, which the CDN serves outside the API quota."""
        self.raw_spent += count

    def observe(self, headers: dict[str, str]) -> None:
        """Record the rate limit a response reported, including when it reaches zero.

        The floor across a run is what matters for planning: a search partition costs one
        request against a limit of ten a minute, so a harvest that ignores the header walks
        into the ceiling having measured nothing about it.
        """
        lowered = {str(key).casefold(): value for key, value in headers.items()}
        limit = lowered.get("x-ratelimit-limit")
        remaining = lowered.get("x-ratelimit-remaining")
        if limit is None and remaining is None:
            return
        if limit is not None:
            self.rate_limit = int(limit)
        if remaining is not None:
            self.rate_remaining = int(remaining)
        reset = lowered.get("x-ratelimit-reset")
        if reset is not None:
            self.rate_reset = int(reset)
        self.rate_resource = str(lowered.get("x-ratelimit-resource", self.rate_resource))
        self._observations.append(
            {
                "resource": self.rate_resource,
                "limit": self.rate_limit,
                "remaining": self.rate_remaining,
                "reset": self.rate_reset,
            }
        )

    def to_dict(self) -> dict[str, object]:
        """Return the budget and the rate limits it observed for the provenance record."""
        return {
            "ceiling": self.ceiling,
            "spent": self.spent,
            "raw_spent": self.raw_spent,
            "ceiling_hits": self.ceiling_hits,
            "stopped": self.stopped,
            "rate_limit": self.rate_limit,
            "rate_remaining": self.rate_remaining,
            "rate_reset": self.rate_reset,
            "rate_resource": self.rate_resource,
            "observations": list(self._observations),
        }


__all__ = ["BudgetExceeded", "RequestBudget"]
