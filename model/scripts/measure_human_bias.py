"""Measure what the served judge reports for repositories no agent could have written.

Validation can only show that the model reproduces its teaching labels. This script asks the
other question: run the same code path the application runs, over repositories whose last
push predates any coding agent, and report the AI percentage it returns. A judge that is
right about authorship should answer near zero on all of them; the distance from zero is the
bias the labels carry, measured on material no label touched.

Dormancy is checked against GitHub's own ``pushed_at`` rather than assumed from a written
list, so a repository that has woken up is skipped instead of silently weakening the claim.

Usage, from the model workspace:

    uv run --project .. python scripts/measure_human_bias.py --limit 20

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import model_paths

model_paths.add_shared_source()

from contracts import parse_report  # noqa: E402
from grading.runtime import Model  # noqa: E402

API = "https://api.github.com/repos/{}"

# Repositories chosen for three properties: a public history that ends before 2023, source
# small enough to grade in one or two evidence batches, and a language the judge sees often
# enough that a high score cannot be blamed on unfamiliar syntax. One line each so a skipped
# entry says why it was skipped.
CANDIDATES: tuple[str, ...] = (
    # C, single-author libraries written long before any agent existed.
    "antirez/sds",
    "antirez/smaz",
    "antirez/linenoise",
    "clibs/list",
    "clibs/hash",
    "clibs/buffer",
    "rxi/map",
    "rxi/vec",
    "rxi/lite",
    "antirez/kilo",
    "orangeduck/mpc",
    "michaelforney/samurai",
    "skeeto/optparse",
    # JavaScript, dormant tooling and the deprecated request client.
    "tj/co",
    "request/request",
    "substack/minimist",
    "harthur/brain",
    "maxogden/websocket-stream",
    "chrisdickinson/beefy",
    "visionmedia/page.js",
    "matthew-andrews/isomorphic-fetch",
    # Python, small hand-written utilities.
    "docopt/docopt",
    "kennethreitz/records",
    "kennethreitz/requests-html",
    # A second pool found by GitHub search rather than by memory (pushed before 2020, 150+
    # stars, small enough to grade): the first pool was mostly still receiving pushes, so
    # the dormancy check discarded it. Kept here rather than fetched live so the sample is
    # reproducible instead of whatever the search ranks that day.
    "fogleman/ln",
    "fatih/structs",
    "pksunkara/alpaca",
    "jakogut/tinyvm",
    "stefanesser/dumpdecrypted",
    "xoreaxeaxeax/rosenbridge",
    "vmg/sundown",
    "subprotocol/verlet-js",
    "zeman/perfmap",
    "Experience-Monks/devtool",
    "petruisfan/node-supervisor",
    "geuis/helium-css",
    "tckmn/mkcast",
    "nryoung/algorithms",
    "vuvuzela/vuvuzela",
    "mailgun/godebug",
    "oklog/oklog",
    "goraft/raft",
    # Held out from the 2026-10-18 training cohort on purpose: they are dormant, so they can
    # measure calibration, and they carry no label, so a checkpoint that trained on the rest
    # has never seen them. `--exclude data/human_repos.txt` leaves these as the post-training check.
    "keendreams/keen",
    "google/eddystone",
    "ngbp/ngbp",
    "fogleman/primitive",
)


def _headers() -> dict[str, str]:
    """Return GitHub request headers, authenticating when a token is configured."""
    headers = {"User-Agent": "HackGrader/0.1", "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def repository_facts(slug: str) -> dict:
    """Read one repository's last push, language, and archived flag from GitHub."""
    request = urllib.request.Request(API.format(slug), headers=_headers())
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    return {
        "pushed_at": payload["pushed_at"],
        "language": payload.get("language"),
        "archived": bool(payload.get("archived")),
        "stars": payload.get("stargazers_count"),
    }


def dormant(pushed_at: str, cutoff: datetime) -> bool:
    """Return whether the repository's last push precedes the cutoff instant."""
    return datetime.fromisoformat(pushed_at.replace("Z", "+00:00")) < cutoff


def grade(grader: Model, slug: str) -> dict:
    """Grade one repository through the application's own query path."""
    started = time.monotonic()
    result = grader.query_with_retry(f"Score this repository: https://github.com/{slug}")
    elapsed = time.monotonic() - started
    parsed = parse_report(result.get("content") or "")
    return {
        "status": result.get("status"),
        "ai_score": parsed["ai_score"],
        "code_score": parsed["code_score"],
        "batches": result.get("batch_count"),
        "seconds": round(elapsed, 1),
        "report": result.get("content"),
    }


def summarize(rows: list[dict]) -> None:
    """Print the distribution a single number would hide."""
    scores = [row["ai_score"] for row in rows if isinstance(row["ai_score"], int)]
    if not scores:
        print("[!] No repository produced a readable AI score.")
        return
    ordered = sorted(scores)
    flagged = sum(1 for score in scores if score >= 40)
    high = sum(1 for score in scores if score >= 70)
    print()
    print(f"[i] {len(scores)} repository report(s) read")
    print(f"[i] mean {statistics.mean(scores):.1f}  median {statistics.median(scores):.0f}")
    print(f"[i] range {ordered[0]}-{ordered[-1]}")
    print(f"[i] at or above the 40% flag: {flagged}/{len(scores)}")
    print(f"[i] at or above 70%: {high}/{len(scores)}")
    print(f"[i] scores: {ordered}")


def main() -> None:
    """Check dormancy on GitHub, grade every surviving repository, and report the spread."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=20, help="repositories to grade")
    parser.add_argument(
        "--cutoff",
        default="2023-01-01",
        help="latest last-push date still treated as pre-agent (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--out", default="reports/human_bias.jsonl", help="result file"
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help=(
            "file of repository URLs to skip, for judging a model that already trained on "
            "them: those repositories measure fit, not calibration"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="keep rows already in --out and grade only the candidates missing from it",
    )
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.fromisoformat(args.cutoff).replace(tzinfo=UTC)

    rows: list[dict] = []
    if args.skip_existing and Path(args.out).exists():
        with open(args.out, encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]
        print(f"[i] keeping {len(rows)} repository result(s) already in {args.out}")
    graded = {row["slug"] for row in rows}
    if args.exclude:
        excluded = {
            line.split("#", 1)[0].strip().removeprefix("https://github.com/").strip("/")
            for line in Path(args.exclude).read_text(encoding="utf-8").splitlines()
        }
        excluded.discard("")
        before = len(graded)
        graded |= excluded
        print(f"[i] excluding {len(graded) - before} repository(ies) from {args.exclude}")
    grader = Model()
    readiness = grader.preflight_error()
    if readiness:
        raise SystemExit(f"[!] {readiness}")

    for slug in CANDIDATES:
        if len(rows) >= args.limit:
            break
        if slug in graded:
            continue
        try:
            facts = repository_facts(slug)
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            print(f"[!] {slug}: could not read GitHub metadata: {exc}")
            continue
        if not dormant(facts["pushed_at"], cutoff):
            print(f"[i] {slug}: skipped, pushed {facts['pushed_at']} on or after {args.cutoff}")
            continue
        print(f"[i] grading {slug} (last push {facts['pushed_at'][:10]})")
        try:
            outcome = grade(grader, slug)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            outcome = {"status": None, "ai_score": None, "content": str(exc)}
        # The tag belongs in the record: without it a later rerun cannot say which adapter
        # produced a score, and a comparison across checkpoints would be guesswork.
        row = {"slug": slug, "model": grader.model, **facts, **outcome}
        rows.append(row)
        print(
            f"[i] {slug}: AI {outcome['ai_score']}% / quality {outcome['code_score']}% "
            f"over {outcome.get('batches')} batch(es) in {outcome.get('seconds')}s"
        )
        with open(args.out, "w", encoding="utf-8") as file:
            for record in rows:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print()
    for row in rows:
        print(
            f"{row['slug']:32s} {row['pushed_at'][:10]}  "
            f"AI {row['ai_score']!s:>3}  quality {row['code_score']!s:>3}"
        )
    summarize(rows)
    print(f"\n[i] wrote {args.out}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
