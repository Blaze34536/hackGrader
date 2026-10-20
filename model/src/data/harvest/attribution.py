"""Turn a file's commit history into an authorship attribution.

The rule is per file and per last touch, because a file is what the grader reads and a
pull request is not: an agent PR that edited 30 lines of a 900-line module authored 3% of
it, while a human PR that rewrote the same module afterwards leaves no agent code behind at
all. So a file is AI-attributed when the *most recent* commit touching it was authored by a
registered agent app, and its attributed line count is what that commit added, capped by the
length the file has now.

The human class is the same rule read the other way, and it is why the touched-commit window
is recorded: a file is called human only when no agent commit appears anywhere in the newest
``window`` commits that touched it. The absence of an agent touch is not proof that no agent
ever wrote the file, so the strong claim — never touched, over this many touches — is the one
the record makes, and a file with an agent commit older than the window is left undecided
rather than counted as human.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from typing import Any

from .agents import AgentApp, agent_of
from .records import AGENT_FILE, HUMAN_FILE, UNKNOWN_FILE, FileTruth

# Commits examined per file. One request returns the newest five touches as easily as the
# newest one, so the human class is decided over a real history rather than a single commit.
TOUCH_WINDOW = 5


def commit_agent(commit: dict[str, Any]) -> AgentApp | None:
    """Return the registered agent that authored a commit, or ``None``.

    ``author`` is the GitHub account GitHub attributed the commit to and is absent for a
    commit it could not attribute, which is why the git author is checked too.
    """
    git_author = commit.get("commit")
    inner = git_author.get("author") if isinstance(git_author, dict) else None
    return agent_of(commit.get("author"), inner)


def commit_date(commit: dict[str, Any]) -> str:
    """Return the ISO date recorded for a commit, preferring the committed date."""
    inner = commit.get("commit")
    if isinstance(inner, dict):
        for key in ("committer", "author"):
            stamp = inner.get(key)
            if isinstance(stamp, dict) and stamp.get("date"):
                return str(stamp["date"])
    return ""


def commit_sha(commit: dict[str, Any]) -> str:
    """Return a commit's SHA."""
    return str(commit.get("sha") or "")


def commit_login(commit: dict[str, Any]) -> str:
    """Return the login GitHub attributed a commit to, or the git author's name."""
    author = commit.get("author")
    if isinstance(author, dict) and author.get("login"):
        return str(author["login"])
    inner = commit.get("commit")
    git_author = inner.get("author") if isinstance(inner, dict) else None
    if isinstance(git_author, dict):
        return str(git_author.get("name") or "")
    return ""


def classify_touches(
    touches: list[dict[str, Any]], *, window: int = TOUCH_WINDOW
) -> tuple[str, AgentApp | None]:
    """Return the attribution of a file and the agent that last touched it, if any.

    ``touches`` is the newest-first history of commits that changed the path.
    """
    if not touches:
        return UNKNOWN_FILE, None
    newest_agent = commit_agent(touches[0])
    if newest_agent is not None:
        return AGENT_FILE, newest_agent
    for commit in touches[:window]:
        if commit_agent(commit) is not None:
            # An agent changed this file, just not most recently. It is neither a file an
            # agent last wrote nor one with no agent history, so it is left out of both
            # classes instead of being handed to the human side.
            return UNKNOWN_FILE, None
    return HUMAN_FILE, None


def file_truth(
    path: str,
    touches: list[dict[str, Any]],
    *,
    window: int = TOUCH_WINDOW,
    blob_bytes: int = 0,
    additions: int = 0,
    current_lines: int = 0,
) -> FileTruth:
    """Build the recorded truth for one file from its commit history."""
    attribution, agent = classify_touches(touches, window=window)
    newest = touches[0] if touches else {}
    return FileTruth(
        path=path,
        attribution=attribution,
        touches_examined=min(len(touches), window),
        blob_bytes=blob_bytes,
        last_touch_sha=commit_sha(newest),
        last_touch_agent=agent.slug if agent else "",
        last_touch_author=commit_login(newest),
        last_touch_date=commit_date(newest),
        additions=additions if attribution == AGENT_FILE else 0,
        current_lines=current_lines,
    )


__all__ = [
    "TOUCH_WINDOW",
    "classify_touches",
    "commit_agent",
    "commit_date",
    "commit_login",
    "commit_sha",
    "file_truth",
]
