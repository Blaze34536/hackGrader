"""Name the AI-share bands a balanced cohort fills, so training sees the whole scale.

A cohort harvested in search order is mostly repositories where one agent pull request
touched a corner of a mature project: the recorded manifest had a median attributed share of
13%, so a percentage target trained on it learned to answer low for anything labeled ``ai``.
Balancing by band gives the judge examples at every part of the scale it reports: verified
human code at 0, partly agent-written projects, and mostly agent-written ones.

The mid band stops below the 40% flag threshold and the high band starts at 50%, so the
two never straddle the line validation classifies on, and a share between them, or one under
10% that is barely distinguishable from human code, fills neither band.

Author: Nihal M
Date: 2026
License: MIT
"""

from __future__ import annotations

AI_MID = "ai-mid"
AI_HIGH = "ai-high"
HUMAN = "human"

# Inclusive lower bound, exclusive upper bound, as a fraction of examined lines.
AI_BANDS = {AI_MID: (0.10, 0.40), AI_HIGH: (0.50, 1.01)}


def band_of(fraction: float | None) -> str | None:
    """Return the AI band a measured share falls in, or ``None`` outside every band."""
    if fraction is None:
        return None
    for band, (low, high) in AI_BANDS.items():
        if low <= fraction < high:
            return band
    return None


def entry_band(entry: dict) -> str | None:
    """Return the band of one label-manifest entry, or ``None`` when it fills none."""
    if entry.get("label") == HUMAN:
        return HUMAN
    examined = int(entry.get("examined_lines") or 0)
    if not examined:
        return None
    return band_of(int(entry.get("attributed_lines") or 0) / examined)


__all__ = ["AI_BANDS", "AI_HIGH", "AI_MID", "HUMAN", "band_of", "entry_band"]
