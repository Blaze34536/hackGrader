# Shared report and evidence contract

The `contracts` package defines the conversation, report, citation, and repository-dump formats used by both live grading and model training. Keeping one contract prevents the application and its trained model from using different instructions or accepting different answers. The live judge, not the detector, owns both project scores.

## Report shape

A valid assistant report has three parts, in this order:

1. An AI-involvement score from 0 to 100 percent, followed by a `- Why this score:` line, then distinct quoted signs ranked strongest-to-weakest; each names its file and explains its weight. If there are no signs, the report says so explicitly.
2. A code-quality score from 0 to 100 percent, with its own `- Why this score:` line and quoted evidence.
3. One short project summary sentence, at most about 25 words.

The justification lines are required by the prompt but are not part of what the parser validates: a report that omits one still parses, so an answer that reasoned about the code is never discarded for a missing sentence. A justification placed inside the ranked-evidence block is skipped by the evidence validator rather than read as a malformed entry.

Both scores, both justifications, and every evidence entry must rest on source code the view reproduces. Only a code file can carry a sign: a README or other prose, a licence, a dependency manifest or lock file (`package.json`, `pyproject.toml`, `requirements.txt`, `go.mod`), and build or CI configuration are what every project of that shape carries, so a sign read only from one of them is not evidence about the author and the contract does not publish it. [`carries_authorship_evidence`](repository_dump.py) states which path categories count, and one statement serves both consumers: `verify_citations` reports a packaging-only entry as unsupported, and `authorship_evidence` drops it from the evidence a report publishes. Formatting is refused for the same reason — indentation and whitespace differ by editor and by language — so the measurement block carries neither count and an entry resting on one is unsupported: the stored gallery run's strongest sign was "0 line(s) are tab indented, for example `using UnityEngine;` in CatClicked.cs", a claim the view no longer measures and a number the judge could not have read there.

Published evidence is the strongest the batches offered rather than the first. A repository is graded as several views, so the aggregate ranks entries by the batch whose score they carried: the batch that scored highest is the one whose evidence decided the repository's number, and each batch keeps its own strongest-first order behind it.

A repository is graded in several evidence batches, so the summary states the limit per batch: the combined report keeps the first usable batch summary instead of concatenating one narrative per batch, which repeated the whole story a dozen times for a split repository.

The parser checks that required scores can be read and that evidence citations are present and supported by the repository view. Every quotation in an evidence entry must occur in the evidence supplied for that report, and every file the entry names must be one that view reproduces: a real quote may not carry an invented file name, and dump bookkeeping (headers, coverage counts, excerpt markers) is not evidence because it says nothing about the repository's content. An entry that quotes nothing is not checkable evidence either. Deductions are exempt, since they price the work rather than describe the code. A plausible but unseen claim is unsupported; runtime consumers preserve the judge's scores and expose unsupported citations separately for review. Detector measurements are factual prompt context only, never scores or authorship evidence.

Two rules keep that check from over-reaching. A quoted token only counts as a cited file name when it is a single path-shaped token, because quoted source routinely carries a slash inside it (`src="songs/theme.mp3"`, a route, a `data:image/jpeg` prefix) and reading those as cited paths rejected evidence the view had genuinely reproduced. And an omission marker -- a truncated excerpt, an omitted file, a redacted encoded payload -- is bookkeeping rather than content, so an entry that quotes one is rejected as unsupported.

Training rows use the same system prompt and report format. Their user message includes one bounded evidence view and the shared report instruction. Every target citation must be present in that row's view. Changes to prompts, score syntax, file markers, batch labels, or citation rules therefore affect both the application and model-data contract tests.

## Model window

One token window is shared by candidate generation and live grading. `MODEL_WINDOW_TOKENS` (`HACKGRADER_MAX_LEN`, default `3072`) is the window training certifies every candidate row against, and `RESPONSE_TOKENS` (`HACKGRADER_NUM_PREDICT`, default `800`) is the reserve the answer may use, so `PROMPT_TOKEN_BUDGET` is the window minus that reserve. `MODEL_DUMP_CHARS` bounds only a single evidence view and cannot bound the whole prompt, which also carries the system prompt, the measurement block, and the question. Serving counts each assembled prompt against the budget before generating, and re-cuts the view when it overflows.

## Repository evidence format

Repository readers emit a header with repository and file-coverage information, followed by file blocks marked with `--- FILE: path ---`. Dumps may omit files or excerpts to fit context limits; the header and omission markers communicate those limits. The model sees bounded batches rather than necessarily the entire repository. A report should describe only the shown evidence.

## Package map

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Exposes the public prompts, markers, parsing, rendering, batching, and aggregation helpers. |
| `prompts.py` | Holds the shared instructions, score patterns, and file markers. |
| `parsing.py` | Parses, renders, and validates reports and citations. |
| `rendering.py` | Builds model-facing messages and evidence summaries. |
| `aggregation.py` | Combines the reports that validated across separate evidence batches and records the ones that did not. |
| `repository_dump.py` | Parses repository headers, file blocks, and coverage counts. |
| `dump_trimming.py` | Selects file content that fits an evidence-character budget. |
| `dump_batching.py` | Splits evidence into bounded, labeled views. |

Import stable helpers from `contracts`, rather than duplicating markers or score parsing in a caller. Before changing the contract, check the [grading runtime](../grading/README.md), then run the offline suite with `uv run pytest` from the repository root; `uv run pytest tests/test_runtime_contracts.py tests/test_detector.py` narrows it to the contract tests.
