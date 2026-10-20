"""Configure import paths for the application tests and report scoring results.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = str(ROOT / "src")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)
# The score targets the training data is built from are part of the same contract the judge
# follows, so the model workspace's `data` package is importable here and their agreement is
# checked by tests rather than by rebuilding a dataset. Importing it adds `src/` itself.
MODEL_SOURCE = str(ROOT / "model" / "src")
if MODEL_SOURCE not in sys.path:
    sys.path.insert(0, MODEL_SOURCE)

# The reports a live scoring run saves, including the summary table it printed.
SCORING_ARTIFACT = ROOT / "build" / "scoring" / "last-run.json"
_session_started = time.time()


def pytest_sessionstart(session: Any) -> None:
    """Note when the session began, so a stale artifact is never reported as this run's."""
    global _session_started
    _session_started = time.time()


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    """Repeat the scores a live scoring run produced, which capture would otherwise hide.

    pytest only shows a passing test's output with ``-s``, and the scores are the whole
    reason to run the live scoring test, so they are printed again in the session summary
    whether or not output was captured. Only an artifact written during this session is
    shown: an offline run must not report the scores of some earlier run.
    """
    try:
        written = SCORING_ARTIFACT.stat().st_mtime
    except OSError:
        return
    if written < _session_started:
        return
    try:
        summary = json.loads(SCORING_ARTIFACT.read_text(encoding="utf-8"))["summary"]
    except (OSError, ValueError, KeyError):
        return
    terminalreporter.write_sep("=", "scoring results")
    terminalreporter.write_line(summary)
