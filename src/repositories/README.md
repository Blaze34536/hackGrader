# GitHub repository access

The `repositories/` package reads public GitHub files and formats selected content as a bounded evidence dump. It does not clone the repository or promise to include every file. The grader and deterministic detector consume the dump; see [grading](../grading/README.md) and [report contracts](../contracts/README.md).

## Public interface

Use `repositories.reader.read_repository(url)` for a synchronous read. It accepts GitHub repository URLs and returns a formatted dump on success or a human-readable error string for invalid URLs, failed reads, or timeouts. `github_reachable()` is a lightweight API reachability check. The implementation runs the asynchronous collector for synchronous callers, including callers that already have an active event loop.

## What is included

The collector selects prioritized source files, applies path and extension exclusions, and records repository identity plus counts of found, fetched, included, omitted, and truncated files where available. The shared dump markers describe included files. Current collection limits are defined in [`settings.py`](settings.py): up to 120 files, 6,000 characters per file, 32,000 total characters of which at most 8,000 may be documentation, and 120 seconds per repository read. These bounds are separate from the smaller per-request evidence window used by the grader.

Read order is what keeps code in the evidence, because the read stops once the total character budget is spent. Every category placed ahead of source takes the budget with it: a large repository keeps a README in dozens of subprojects, and ranking all of them first once spent the entire budget and returned a dump with no source code at all. The collector therefore reads source, entry points, and tests first, then the project's metadata and configuration, then the root README, then every other document, and nested READMEs last, bounded by the documentation budget. The README sits behind the code because it is the project's own account of itself: a read that leads with it hands the grader a description to score instead of a repository. `view_priority` in [`contracts/repository_dump.py`](../contracts/repository_dump.py) states that category order once, `selection_priority` in [`selection.py`](selection.py) reads it for the fetch order, and `is_documentation` distinguishes prose from source.

Trained model weights, media, archives, and other binary artifacts stay out of the evidence. Unambiguous formats are excluded by extension, and the bytes actually fetched are checked as well, so an extensionless or mislabelled blob is dropped too. Every excluded path is listed under `FILES NOT INCLUDED` and spends no character budget, so the omitted bytes leave room for source files instead of consuming the view.

Encoded payloads inside readable files are a second form of the same hazard. An inline `data:image/jpeg;base64,...` icon or a base64 banner is valid text, so the byte check passes it, and a grader served one of those quotes its characters back as authorship evidence: one graded project's stylesheet carried a 5,800-character JPEG as a single line, and the report cited bytes from it. `redact_payloads` in [`selection.py`](selection.py) replaces any run of 120 or more base64 characters with `[... encoded payload omitted: N character(s) ...]`, so the payload cannot be quoted, is not scored, and stops spending the per-file budget before the code does. The marker is bookkeeping, so a citation that quotes it is rejected like any other dump header.

## What happens when GitHub truncates the tree

The collector discovers files from `git/trees/<branch>?recursive=1`, which GitHub truncates for very large repositories. A truncated listing is a partial one, and where GitHub stops can hide everything the project wrote: one graded repository kept only `.gitignore` while 37k listed paths sat under a Cloud SDK the selector ignores, and the judge then scored 85% AI involvement from eight lines of `.gitignore`.

So when the tree reports `truncated`, the collector ignores the partial listing and lists directories through the contents API instead, starting at the root and going shallowest-first. The walk keeps the paths the tree did return, skips ignored directories, and stops at 40 listings or four levels deep (`MAX_TREE_REQUESTS` and `MAX_TREE_DEPTH` in [`settings.py`](settings.py)), which reaches files five components down such as `src/components/ui/panel.tsx`. Each listing costs one GitHub API request, so an untruncated tree spends none: the walk is a fallback, not the normal path.

If the tree is truncated and the root listing cannot be read — a rate limit, a network failure — the read returns an error rather than a dump. Scoring the repository from whatever stray file the tree happened to list produces a confident, wrong percentage; a failed read is reported as a failure and can be retried. `FILES FOUND` counts what discovery actually found, and `FILES TRUNCATED: yes` still records that neither listing was complete.

A GitHub personal access token is optional. Set `GITHUB_TOKEN` in the process environment to raise API limits. Do not place tokens in documentation, fixtures, or committed source. Public repositories can still be unavailable because of rate limits, network failures, repository changes, or access restrictions.

## Package map

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Package marker with no re-exports; import `repositories.reader` for the public entry point. |
| `urls.py` | Validates GitHub URLs and normalizes repository slugs. |
| `settings.py` | Defines fetch limits and ignored paths/extensions. |
| `selection.py` | Chooses files, detects binary bodies, redacts encoded payloads, and bounds path listings. |
| `discovery.py` | Lists directories, bounded, when GitHub truncates the recursive tree. |
| `transport.py` | Fetches files with timeouts and retries. |
| `collection.py` | Coordinates reads and creates headers, file blocks, and coverage notes. |
| `reader.py` | Provides the public synchronous entry point. |

Keep file counts and omission markers accurate when changing collection behavior; report citations and coverage rely on them. Network reads in tests should be mocked to avoid rate limits and nondeterministic repository contents.
