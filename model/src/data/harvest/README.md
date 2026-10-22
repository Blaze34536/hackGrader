# Cohort harvest reference

Repository authorship has to come from a record somebody else can recheck. This package
builds that record from the GitHub API: merged pull requests authored by registered coding
agents, the commit that last touched every file the reader would read, and the label manifest
those commits support under the fail-closed standard in [`../labels.py`](../labels.py).

It exists because the alternative was measured and failed. On the 52-repository pool in
`../../../data/repos.fresh.txt` no repository satisfies that standard — 8 declare AI authorship in
their own files, 44 offer no authorship evidence — and the style reading the repository head
was trained on put human projects at a mean position of 0.71 against 0.62 for AI-declared
ones. A cohort of agent-authored pulls is the independent measurement that replaces it.

## Run it

```bash
# Plan a run without spending a request.
uv run src/cli.py harvest --since 2026-09-01 --until 2026-09-30 --dry-run

# Harvest one month of agent work into a resumable cohort, then write the label manifest.
uv run src/cli.py harvest --since 2026-09-01 --until 2026-09-30 \
  --per-agent-limit 25 --max-repositories 60 \
  --out data/cohort.jsonl --labels-out data/labels.audited.jsonl \
  --human-repositories data/repos.fresh.txt --human-before 2023-01-01

# Supervise a data build from labels that passed the standard, not from a style reading.
# The helper derives data/candidate_repos.txt from the manifest and validates it first.
./scripts/build_candidate_from_harvest.sh
```

`GITHUB_TOKEN` is optional and decisive: an authenticated run may spend 5,000 core requests
an hour, and an anonymous one 60 core and 10 search requests an hour. The run prints which
limit it is operating under and records the limit GitHub reported.

`--resume-to-goal` builds the cohort in one invocation when one budget cannot reach
`--max-repositories`. Each pass gets a fresh `--request-budget`; when the hourly quota is
exhausted the command sleeps until the reported `X-RateLimit-Reset`, then continues. It stops
early when a pass records no new repository, when no reset was ever reported, or after
`--max-runs` passes, and it exits non-zero unless the goal was reached. Combining it with
`--refresh` makes re-reads count toward the goal, so use the two together deliberately.

Rate limits are pauses, not missing resources. A `429`, or a `403` carrying `Retry-After`,
`X-RateLimit-Remaining: 0`, or a rate-limit message, is honored for its full `Retry-After`
(seconds or HTTP date) or reset time; waits up to two minutes are absorbed inline and retried.
A longer pause stops the pass with its `retry_after` recorded, which `--resume-to-goal` waits
out before the next budget; a single-pass run prints the pause and exits non-zero. An ordinary
`403` such as a private repository stays a miss and is not retried.

| Option | Default | Effect |
| --- | ---: | --- |
| `--agents` | all registered | Comma-separated registered agent slugs to search. |
| `--since` / `--until` | last complete month | Merged-date window, partitioned by calendar month. |
| `--per-agent-limit` | `25` | Merged pull requests read per agent. |
| `--max-repositories` | `60` | Repositories the cohort should reach; the design floor is 60. |
| `--out` | `data/cohort.jsonl` | Cohort file, appended to and resumed from. |
| `--labels-out` | none | Write the fail-closed label manifest here. |
| `--human-repositories` | none | File of repository URLs to verify as the human class. |
| `--human-before` | `2023-01-01` | Date every commit of a human-class repository must predate. |
| `--human-search` | `0` | Human-class candidates to find by repository search (pushed before `--human-before`), verified like the list. |
| `--band-goal` | off | Repositories wanted per AI band (10–40% and ≥50% attributed). The run stops once both bands hold N, and the manifest keeps at most this many per band, human included. Replaces `--max-repositories` as the goal. |
| `--request-budget` | `5000` | Metered requests this run may spend. |
| `--resume-to-goal` | off | Keep harvesting with a fresh budget until `--max-repositories` is reached. |
| `--max-runs` | `4` | Harvest passes `--resume-to-goal` may spend. |
| `--max-files-per-repository` | `40` | Files attributed per repository, ranked as the reader ranks them. |
| `--touch-window` | `5` | Commits examined per file, which decides the human class. |
| `--pause` | `0` | Seconds before each metered request, for anonymous search limits. |
| `--refresh` | off | Re-harvest repositories the cohort already holds. |

Every setting also has a `HACKGRADER_HARVEST_*` environment override, listed in the
[data reference](../README.md).

## The attribution rule (`last-touch-v1`)

> A file is **AI-attributed** when the most recent commit touching it was authored by a
> registered agent app. Its AI line count is the number of lines that commit added to it,
> bounded by the file's current length.

Preferring the last touch over a pull request's whole diff is deliberate: a PR that edits 30
lines of a 900-line module authored 3% of it, and a human who rewrote the file afterwards
leaves no agent code in it at all. The complementary class is a file with **no** agent commit
among the newest `--touch-window` commits that touched it; a file whose agent commit sits
further back is recorded as `unknown` and counted in neither class, because "no agent touch
observed" is not the same claim as "no agent wrote this".

Three signals identify an agent commit, strongest first: the app page GitHub associates with
the account (`https://github.com/apps/<slug>`), a registered login or commit identity, and the
git author name or address. The registry is not "anything of type `Bot`": dependabot merging
a lockfile bump is not authorship of the code around it.

Which files get read follows from the same measurement. The pull request's own paths are
examined first and the reader's ranking fills the rest, because a ranking alone missed the
agent entirely: one live sample's pull request changed a log document and three SQL migrations
while the top-ranked paths were the project's own modules, so five files the agent never
touched were examined and an agent-written repository looked human. Ordering the read with the
pull request does not weaken the rule — the attribution still comes from the last-touch commit,
not from the diff.

The registered apps and their merged-PR counts were measured live on 2026-10-15, because a
wrong slug is not a smaller cohort but a silent one — `author:app/<unknown>` answers HTTP 422
while a plausible guess such as `devin` fails the same way:

| Slug | Pull-request author | Merged PRs | Commit identity (read from a merged PR) |
| --- | --- | ---: | --- |
| `copilot-swe-agent` | `Copilot` | 1,538,405 | `Copilot`, `copilot-swe-agent[bot]`, `…+Copilot@users.noreply.github.com` |
| `google-labs-jules` | `google-labs-jules[bot]` | 204,937 | `google-labs-jules[bot]` and its no-reply address |
| `devin-ai-integration` | `devin-ai-integration[bot]` | 191,813 | **none** — its commits carry the human's own account |
| `cursor` | `cursor[bot]` | 30,601 | `cursoragent` / "Cursor Agent" / `cursoragent@cursor.com` |
| `codegen-sh` | `codegen-sh[bot]` | 4,007 | not sampled |

The two columns are why the registry carries identities rather than one login. Cursor's pull
request author is the app while its commits are authored by a plain account, so a rule that
knew only `cursor[bot]` classified every Cursor-written file as untouched by any agent — the
dangerous direction, since it hands agent code to the human control class. Devin is the
measured opposite: its pull requests are real evidence, its commits identify the human, so its
repositories are labelable from the pull request and merge commit but can never supply
file-level attribution, and the harvest says so instead of guessing.

## What a run records

| File | Contents |
| --- | --- |
| `data/cohort.jsonl` | One record per repository: the agent pull request and merge commit, every examined file with its last-touch SHA, author, attribution, additions, and current length, plus the requests spent. |
| `cohort.provenance.json` | One entry per run, or per pass under `--resume-to-goal`: the search queries, their result counts and pages read, the rate limit observed, the budget spent, the languages reached, and every human-class rejection with its reason. |
| the `--labels-out` manifest | `ai` entries carrying the pull request and merge commit, and `human` entries carrying the protocol that ruled the class in. |

The cohort file is appended to and fsynced per repository, so a run that loses its budget or
its network keeps every repository it finished, and re-running the same command skips what is
already recorded. A torn final line from an interrupted write is dropped on read rather than
failing the run.

## What it does not establish

The attributed fraction in a record is over the files the harvest examined, not over the
repository: `--max-files-per-repository` bounds the read, and the reader's own coverage is
reported separately when the builder reads the same repository. The design's cohort floor —
60 repositories, at least 3 languages, both classes present — is reported by the run and not
enforced by it, so a small first harvest is visible as small rather than refused. Nothing
here asserts that a repository with no agent commit was written without AI: that is what the
recorded protocol and its commit sample are for, and a repository that fails any part of it
is rejected with its reason. Nor does every agent supply file-level truth: Devin's commits
carry the human's identity, so a Devin repository is recorded with its pull request and merge
commit and no attributed files, which the harvest prints rather than hides. Only the first page
of a pull request's files is read, so a pull request changing more than a hundred files has its
first hundred ordered ahead of the rest.
