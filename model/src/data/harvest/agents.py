"""The registered agent apps whose merged pull requests make a repository cohort-eligible.

Every entry here was verified against the live API before it was written down, because a slug
that does not exist does not merely return nothing: ``author:app/<slug>`` answers HTTP 422 for
an unknown app, while a *plausible* wrong slug such as ``devin`` fails the same way and a
cohort would silently be built from one agent.

Matching an agent's work needs two identities, and they are not the same one. The pull request
author is the app (``cursor[bot]``), but the commits inside that pull request are authored by
whatever identity the agent commits under, and for Cursor that is a plain user account,
``cursoragent``, whose commits are named "Cursor Agent". A registry that knew only the pull
request author classified every Cursor-authored file as untouched by any agent, which is the
dangerous direction: it hands agent-written code to the human control class. The commit
identities are recorded per agent for that reason.

Devin is the measured exception. Its merged pull requests are real — 191,813 of them — but the
commits sampled inside them were authored by the human's own account
(``good2better`` / a personal address), so no commit of Devin's can be recognised as agent
authorship. Its repositories are still labelable from the pull request and merge commit, and
they are marked as supplying no file-level attribution, which is what the harvest reports
rather than guessing.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

APPS_PREFIX = "https://github.com/apps/"


@dataclass(frozen=True)
class AgentApp:
    """One registered coding agent: how its pull requests and its commits are identified."""

    slug: str
    login: str
    name: str
    commit_logins: tuple[str, ...] = ()
    commit_names: tuple[str, ...] = ()
    commit_emails: tuple[str, ...] = ()
    commit_attributable: bool = True

    @property
    def commit_identities(self) -> set[str]:
        """Return every casefolded login and name that identifies this agent's commits."""
        identities = {self.login.casefold(), self.slug.casefold(), f"{self.slug.casefold()}[bot]"}
        for value in (*self.commit_logins, *self.commit_names):
            identities.add(value.casefold())
            identities.add(value.casefold().removesuffix("[bot]"))
        return identities

    def matches_commit_email(self, email: str) -> bool:
        """Return whether an address is one of this agent's recorded commit addresses.

        GitHub's agent addresses are of the form ``<id>+<name>@users.noreply.github.com``, so
        the local part after ``+`` is compared as well: it is the identity GitHub itself uses
        to attribute the commit to an app.
        """
        lowered = email.strip().casefold()
        if not lowered:
            return False
        if lowered in {value.casefold() for value in self.commit_emails}:
            return True
        local = lowered.split("@", 1)[0].rsplit("+", 1)[-1]
        return bool(local) and local in self.commit_identities


# Measured 2026-10-15 with `author:app/<slug>` merged-PR counts, which are recorded here so a
# later reader can tell a real filter from an ignored parameter: copilot-swe-agent 1,538,405 ·
# google-labs-jules 204,937 · devin-ai-integration 191,813 · cursor 30,601 · codegen-sh 4,007.
# The commit identities were read from the commits of one merged pull request per agent.
AGENT_APPS: tuple[AgentApp, ...] = (
    AgentApp(
        "copilot-swe-agent",
        "Copilot",
        "GitHub Copilot coding agent",
        commit_logins=("Copilot",),
        commit_names=("copilot-swe-agent[bot]",),
        commit_emails=("198982749+Copilot@users.noreply.github.com",),
    ),
    AgentApp(
        "google-labs-jules",
        "google-labs-jules[bot]",
        "Jules",
        commit_emails=("161369871+google-labs-jules[bot]@users.noreply.github.com",),
    ),
    AgentApp(
        "devin-ai-integration",
        "devin-ai-integration[bot]",
        "Devin",
        commit_attributable=False,
    ),
    AgentApp(
        "cursor",
        "cursor[bot]",
        "Cursor",
        commit_logins=("cursoragent",),
        commit_names=("Cursor Agent",),
        commit_emails=("cursoragent@cursor.com",),
    ),
    AgentApp("codegen-sh", "codegen-sh[bot]", "Codegen"),
)

AGENT_SLUGS: tuple[str, ...] = tuple(app.slug for app in AGENT_APPS)

_BY_SLUG = {app.slug.casefold(): app for app in AGENT_APPS}
_BY_LOGIN: dict[str, AgentApp] = {}
for _app in AGENT_APPS:
    for _identity in _app.commit_identities:
        _BY_LOGIN.setdefault(_identity, _app)


def agent_for_slug(slug: str) -> AgentApp | None:
    """Return the registered agent for an app slug, or ``None`` when it is not registered."""
    return _BY_SLUG.get(slug.strip().casefold())


def agent_for_login(login: str) -> AgentApp | None:
    """Return the registered agent whose pull-request or commit login matches, or ``None``."""
    return _BY_LOGIN.get(login.strip().casefold())


def agent_for_app_url(url: str) -> AgentApp | None:
    """Return the registered agent named by a GitHub app page URL, or ``None``."""
    if not url.startswith(APPS_PREFIX):
        return None
    return agent_for_slug(url[len(APPS_PREFIX):].strip("/"))


def _agent_for_commit_author(author: dict[str, Any] | None) -> AgentApp | None:
    """Match a commit's git author name or address against the registry."""
    if not isinstance(author, dict):
        return None
    name = str(author.get("name") or "").strip().casefold()
    email = str(author.get("email") or "").strip()
    for app in AGENT_APPS:
        if name and name in app.commit_identities:
            return app
        if app.matches_commit_email(email):
            return app
    return None


def agent_of(
    user: dict[str, Any] | None, commit_author: dict[str, Any] | None = None
) -> AgentApp | None:
    """Return the registered agent that authored a commit, or ``None`` for anybody else.

    Signals are tried strongest first: the app page GitHub associates with the account, then
    the account login or a recorded commit identity, then the git author. A bot account that
    is *not* registered stays unregistered on purpose — dependabot merging a lockfile bump is
    not authorship of the code around it, and treating any ``type: Bot`` commit as AI would
    put a maintenance bot at the top of every attribution table.
    """
    if isinstance(user, dict):
        if app := agent_for_app_url(str(user.get("html_url") or "")):
            return app
        if app := agent_for_login(str(user.get("login") or "")):
            return app
    return _agent_for_commit_author(commit_author)


__all__ = [
    "AGENT_APPS",
    "AGENT_SLUGS",
    "AgentApp",
    "agent_for_app_url",
    "agent_for_login",
    "agent_for_slug",
    "agent_of",
]
