"""Run a cohort harvest from the command line and report what it recorded.

The command is deliberately resumable rather than one-shot: `--dry-run` costs nothing and
prints the partitions a run would search, an interrupted run keeps every repository it
finished, and re-running the same command skips what the cohort already holds. The label
manifest is written only when asked for, and always through the shared fail-closed validator,
so a run that harvests one class cannot look like a run that harvested two.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

from ..config import (
    HARVEST_BUDGET,
    HARVEST_COHORT_FILE,
    HARVEST_HUMAN_BEFORE,
    HARVEST_MAX_FILES,
    HARVEST_PAUSE_SECONDS,
    HARVEST_TOUCH_WINDOW,
)
from .agents import AGENT_SLUGS, AgentApp, agent_for_slug
from .budget import RequestBudget
from .client import GithubClient, GithubError, github_token
from .cohort import harvest, language_of
from .human import COMMIT_SAMPLE, read_url_list, verify_human_repositories
from .manifest import AUDITOR, derive_entries, report, write_manifest
from .queries import month_windows, partition_query
from .records import CohortStore, read_cohort, write_provenance

MIN_LANGUAGES = 3


def _previous_month() -> tuple[str, str]:
    """Return the last complete calendar month, the default harvest window."""
    first_of_month = date.today().replace(day=1)
    end = first_of_month - timedelta(days=1)
    return end.replace(day=1).isoformat(), end.isoformat()


def _parser() -> argparse.ArgumentParser:
    """Create command-line options for one harvest run."""
    since, until = _previous_month()
    parser = argparse.ArgumentParser(
        prog="uv run src/cli.py harvest",
        description=(
            "Harvest the agent-attributed repository cohort: merged agent pull requests, "
            "per-file last-touch attribution, and the fail-closed label manifest they support."
        ),
    )
    parser.add_argument(
        "--agents",
        default=",".join(AGENT_SLUGS),
        help="comma-separated registered agent slugs to harvest (default: all of them)",
    )
    parser.add_argument("--since", default=since, help=f"first merged date (default: {since})")
    parser.add_argument("--until", default=until, help=f"last merged date (default: {until})")
    parser.add_argument(
        "--per-agent-limit",
        type=int,
        default=25,
        help="merged pull requests to read per agent (default: 25)",
    )
    parser.add_argument(
        "--max-repositories",
        type=int,
        default=60,
        help="repositories the cohort should reach; the design floor is 60 (default: 60)",
    )
    parser.add_argument(
        "--out",
        default=HARVEST_COHORT_FILE,
        help=f"cohort JSONL file, appended to and resumable (default: {HARVEST_COHORT_FILE})",
    )
    parser.add_argument(
        "--labels-out",
        default=None,
        help=(
            "write the fail-closed repository label manifest here, for "
            "`data -r data/repos.txt --repository-labels <file>`; no manifest is written "
            "unless this is given"
        ),
    )
    parser.add_argument(
        "--human-repositories",
        default=None,
        help="file of repository URLs to verify as the human class (for example data/repos.fresh.txt)",
    )
    parser.add_argument(
        "--human-before",
        default=HARVEST_HUMAN_BEFORE,
        help=(
            "date every commit of a human-class repository must predate "
            f"(default: {HARVEST_HUMAN_BEFORE})"
        ),
    )
    parser.add_argument(
        "--human-sample",
        type=int,
        default=COMMIT_SAMPLE,
        help=f"commits inspected per human-class repository (default: {COMMIT_SAMPLE})",
    )
    parser.add_argument(
        "--request-budget",
        type=int,
        default=HARVEST_BUDGET,
        help=(
            "metered GitHub requests this run may spend; a token allows 5000 an hour "
            f"(default: {HARVEST_BUDGET})"
        ),
    )
    parser.add_argument(
        "--max-files-per-repository",
        type=int,
        default=HARVEST_MAX_FILES,
        help=(
            "files attributed per repository, ranked as the reader ranks them "
            f"(default: {HARVEST_MAX_FILES})"
        ),
    )
    parser.add_argument(
        "--touch-window",
        type=int,
        default=HARVEST_TOUCH_WINDOW,
        help=(
            "commits examined per file, deciding the human class "
            f"(default: {HARVEST_TOUCH_WINDOW})"
        ),
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=HARVEST_PAUSE_SECONDS,
        help=(
            "seconds to wait before each metered request, for an anonymous search limit of "
            f"10 a minute (default: {HARVEST_PAUSE_SECONDS})"
        ),
    )
    parser.add_argument(
        "--audited-by",
        default=AUDITOR,
        help=f"name recorded as the auditor in the label manifest (default: {AUDITOR})",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-harvest repositories the cohort already holds instead of skipping them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the partitions this run would search and the cohort's current contents",
    )
    return parser


def _resolve_agents(value: str) -> tuple[AgentApp, ...]:
    """Resolve requested agent slugs, refusing an unregistered one outright.

    GitHub answers HTTP 422 for an unknown app slug rather than returning nothing, but a
    typo caught here costs no requests at all and cannot leave a cohort that silently covers
    one agent.
    """
    agents: list[AgentApp] = []
    for slug in (part.strip() for part in value.split(",")):
        if not slug:
            continue
        agent = agent_for_slug(slug)
        if agent is None:
            registered = ", ".join(AGENT_SLUGS)
            raise SystemExit(f"[!] {slug!r} is not a registered agent app. Registered: {registered}")
        agents.append(agent)
    if not agents:
        raise SystemExit("[!] No agent apps requested.")
    return tuple(agents)


def _languages(cohort) -> dict[str, int]:
    """Count examined files by language across the cohort."""
    counts: dict[str, int] = {}
    for truth in cohort.repositories:
        for item in truth.files:
            name = language_of(item.path)
            counts[name] = counts.get(name, 0) + 1
    return counts


def _dry_run(args, store: CohortStore) -> int:
    """Print the plan and the cohort's current contents without spending a request."""
    cohort = read_cohort(store.path)
    agents = _resolve_agents(args.agents)
    windows = month_windows(args.since, args.until)
    print(f"[i] {len(windows)} partition window(s) from {args.since} to {args.until}")
    for window in windows:
        for agent in agents:
            print(f"    {window[0]}..{window[1]}  {partition_query(agent, window)}")
    counts = cohort.counts()
    print(
        f"[i] cohort {store.path}: {counts['repository']} repository(ies), "
        f"{counts['human']} verified human repository(ies), "
        f"{len(cohort.urls())} URL(s) covered"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one harvest, write its provenance, and optionally write the label manifest."""
    args = _parser().parse_args(argv)
    agents = _resolve_agents(args.agents)
    store = CohortStore(args.out)
    if args.dry_run:
        return _dry_run(args, store)

    token = github_token()
    budget = RequestBudget(ceiling=args.request_budget)
    client = GithubClient(budget, token=token, pause=args.pause)
    if not token:
        print(
            "[i] No GITHUB_TOKEN found; harvesting anonymously at 60 core and 10 search "
            "requests an hour. Set GITHUB_TOKEN for the 5,000-request limit."
        )
    started = date.today().isoformat()
    try:
        summary = harvest(
            client,
            agents=agents,
            since=args.since,
            until=args.until,
            per_agent_limit=args.per_agent_limit,
            max_repositories=args.max_repositories,
            store=store,
            window=args.touch_window,
            max_files=args.max_files_per_repository,
            refresh=args.refresh,
        )
    except GithubError as exc:
        raise SystemExit(f"[!] {exc}") from None

    cohort = read_cohort(store.path)
    human_records = []
    human_rejections: list[tuple[str, str]] = []
    if args.human_repositories:
        urls = read_url_list(args.human_repositories)
        fresh = [url for url in urls if url not in cohort.urls() or args.refresh]
        print(f"[i] verifying {len(fresh)} human-class candidate(s) from {args.human_repositories}")
        human_records, human_rejections = verify_human_repositories(
            client, fresh, cutoff=args.human_before, sample=args.human_sample
        )
        for record in human_records:
            store.add_human(record)
        cohort = read_cohort(store.path)

    languages = _languages(cohort)
    provenance = write_provenance(
        store.path,
        {
            "started": started,
            "finished": date.today().isoformat(),
            "agents": [agent.slug for agent in agents],
            "since": args.since,
            "until": args.until,
            "per_agent_limit": args.per_agent_limit,
            "max_repositories": args.max_repositories,
            "max_files_per_repository": args.max_files_per_repository,
            "touch_window": args.touch_window,
            "human_repositories": args.human_repositories or "",
            "human_before": args.human_before,
            "human_sample": args.human_sample,
            "authenticated": bool(token),
            "stopped": summary.stopped,
            "partitions": [item.to_dict() for item in summary.partitions],
            "budget": budget.to_dict(),
            "errors": summary.errors,
            "cohort": cohort.counts(),
            "languages": languages,
            "human_rejections": [
                {"url": url, "reason": reason} for url, reason in human_rejections
            ],
        },
    )
    print(
        f"[i] cohort {store.path}: {len(cohort.repositories)} repository(ies) with "
        f"attribution, {len(cohort.humans)} verified human repository(ies), "
        f"{budget.spent} metered request(s) spent of {budget.ceiling}"
    )
    if summary.stopped:
        print(f"[i] stopped: {summary.stopped}")
    if languages:
        shown = ", ".join(
            f"{name} {count}" for name, count in sorted(
                languages.items(), key=lambda item: (-item[1], item[0])
            )[:8]
        )
        print(f"[i] languages: {shown}")
        if len(languages) < MIN_LANGUAGES:
            print(
                f"[!] {len(languages)} language(s) seen; the cohort floor is "
                f"{MIN_LANGUAGES} plus {len(cohort.humans)} human-class repository(ies) "
                "and 60 repositories."
            )
    print(f"[i] provenance: {provenance}")

    if args.labels_out:
        entries, dropped = derive_entries(
            cohort, audited_by=args.audited_by, audited_at=date.today().isoformat()
        )
        path = write_manifest(args.labels_out, entries)
        manifest = report(path, entries)
        for url, reason in dropped:
            print(f"[!] no label entry for {url}: {reason}")
        if not manifest.accepted:
            print("[!] The manifest holds no entry the fail-closed standard accepts.")
    if not cohort.repositories and not cohort.humans:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
