"""Locate application modules shared with the separate Python 3.12 model environment.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import sys
from pathlib import Path

SHARED_SOURCE = Path(__file__).resolve().parents[2] / "src"


def add_shared_source() -> None:
    """Make the canonical detector and report contracts importable by model commands."""
    source = str(SHARED_SOURCE)
    if source not in sys.path:
        sys.path.insert(0, source)
