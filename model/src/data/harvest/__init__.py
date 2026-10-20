"""Harvest the agent-attributed repository cohort and the labels it can defend.

The package answers one question with evidence instead of inference: which files in a
repository did a registered coding agent write? It reads merged agent pull requests, resolves
each file's most recent touching commit, records the commit SHAs and the rule version that
decided the attribution, and derives the fail-closed repository label manifest that
``data -r data/repos.txt --repository-labels`` consumes.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from .cli import main

__all__ = ["main"]
