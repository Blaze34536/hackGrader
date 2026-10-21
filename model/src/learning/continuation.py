"""Choose the one safe next step for a training run.

Resume, fork, or rebuild a release? A run that never claimed its test and still has
budget resumes, one whose adapter postdates the claim rebuilds its release, and one that
spent its test without saving an adapter starts over. This module applies that rule from
disk facts alone: it loads no model and writes only the run id it pins.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from .run_artifacts import (
    ATTEMPT_MARKER,
    RELEASED_MODEL,
    RunState,
    continuable_run,
    inspect,
    newest_claimed_run,
    newest_run,
    outputs_root,
)

__all__ = [
    "ATTEMPT_MARKER",
    "Decision",
    "choose",
    "continuable_run",
    "inspect",
    "newest_claimed_run",
    "newest_run",
    "plan",
]


@dataclass(frozen=True)
class Decision:
    """The one continuation action that is safe for a run."""

    action: str
    run_id: str
    checkpoint: str | None
    message: str
    code: int
    inputs: tuple[str, ...] = ()


def plan(
    run_id: str | None = None,
    *,
    root: str | None = None,
    released: str = RELEASED_MODEL,
) -> Decision:
    """Pick the run, pin its id, and decide the safe continuation action."""
    root = root or outputs_root()
    selected = run_id or os.environ.get("HACKGRADER_RUN_ID") or continuable_run(root)
    if selected is None:
        return Decision(
            "empty",
            "",
            None,
            "[!] No run directory with run metadata was found; start one with `train`. "
            "Starting a run is the only way to get its own independent test.",
            1,
        )
    if os.path.basename(selected) != selected:
        return Decision(
            "empty",
            selected,
            None,
            f"[!] {selected!r} is not a bare run id of the form YYYYMMDD-HHMMSS.",
            1,
        )
    if not os.path.isfile(os.path.join(root, selected, "run_metadata.json")):
        return Decision(
            "empty",
            selected,
            None,
            f"[!] {root}/{selected} is not a run directory; pass --run-id with a run "
            "that has run_metadata.json.",
            1,
        )
    # Configuration derives every artifact path from the run id at import time, so the
    # id has to be pinned before the first import of ``learning.config``.
    os.environ["HACKGRADER_RUN_ID"] = selected
    from . import config

    return choose(
        inspect(
            selected,
            root=root,
            adapter_dir=config.ADAPTER_DIR,
            merge_dir=config.MERGE_DIR,
            released=released,
        ),
        hours=config.HOURS,
        reserves_hours=(
            config.VALIDATION_SCORING_RESERVE_HOURS + config.FINAL_TEST_RESERVE_HOURS
        ),
    )


def choose(
    state: RunState,
    *,
    hours: float,
    reserves_hours: float,
    now: float | None = None,
) -> Decision:
    """Return the single safe continuation action for one run's state."""
    now = time.time() if now is None else now
    if state.claimed_at is None:
        return _before_claim(state, hours=hours, reserves_hours=reserves_hours, now=now)
    if (
        state.adapter_file is not None
        and state.adapter_written_at is not None
        and state.adapter_written_at >= state.claimed_at
    ):
        if state.deployed:
            return Decision(
                "complete",
                state.run_id,
                None,
                f"[i] run {state.run_id} is already released; nothing to continue.",
                0,
            )
        return Decision(
            "recover",
            state.run_id,
            None,
            f"[i] run {state.run_id} claimed and passed its independent test, then "
            "stopped before the release finished; rebuilding only the merge and "
            "registration, without scoring the test again.",
            0,
            state.inputs,
        )
    return Decision(
        "spent",
        state.run_id,
        None,
        f"[!] run {state.run_id} spent its independent test and saved no adapter, so "
        "there is no release to rebuild. Start a new run with `train --fork-checkpoint "
        "<checkpoint>`; the next run reserves its own test.",
        1,
    )


def _before_claim(
    state: RunState, *, hours: float, reserves_hours: float, now: float
) -> Decision:
    """Choose between resuming training and reporting why it cannot be resumed."""
    if not state.checkpoints:
        return Decision(
            "empty",
            state.run_id,
            None,
            f"[!] run {state.run_id} has no saved checkpoint, so training cannot "
            "continue from it. Start a new run with `train`.",
            1,
        )
    if state.started_at is None:
        return Decision(
            "empty",
            state.run_id,
            None,
            f"[!] {state.run_dir}/run_metadata.json has no start time, so the original "
            "training cap cannot be enforced. Start a new run with `train`.",
            1,
        )
    remaining = state.started_at + hours * 3600 - now
    if remaining <= reserves_hours * 3600:
        return Decision(
            "expired",
            state.run_id,
            None,
            f"[!] run {state.run_id} has less than {reserves_hours:.1f}h left on its "
            f"original {hours:g}-hour cap, which is reserved for validation and the "
            "independent test. Continue in a new run with `train --fork-checkpoint "
            f"{state.checkpoints[-1][1]}` so it gets its own cap and test.",
            1,
        )
    step, path = state.selected or state.checkpoints[-1]
    return Decision(
        "resume",
        state.run_id,
        path,
        f"[i] run {state.run_id} has not claimed its independent test; resuming from "
        f"checkpoint-{step} with {remaining / 3600:.1f}h left on the original cap",
        0,
        state.inputs,
    )
