# Candidate data reference

The `data/` package builds reviewable training examples from labeled code, public repositories, synthetic projects, and explicit-artifact examples. It preserves source lineage so related material cannot cross the training, validation, and independent-test boundaries.

## Build path

1. `examples.py` loads labeled AIGCodeSet samples and their problem/content lineage.
2. `sources.py` reads repository dumps from the configured cache or fetcher.
3. `synthetic.py` creates projects and artifact examples linked to source examples.
4. `features.py`, `evidence.py`, and `rendering.py` create scores, supporting citations, and bounded chat rows. An evidence entry states the measurement the target rests on and quotes a line the view really contains; it never opens by declaring the code machine-written or hand-written, because that sentence is the one claim the view cannot check and the trained model reproduced it for clearly human code. Only a quoted artifact names an authorship.
5. `rows.py` removes conflicting prompts, merges safe duplicates, and caps score buckets.
6. `builder/` allocates whole source components and publishes only after checks pass.

See the [builder reference](builder/README.md) for CLI options and the [splitting reference](splitting/README.md) for allocation behavior.

## Candidate row schema

Each JSONL line is one object with:

| Field | Meaning |
| --- | --- |
| `messages` | Three objects in `system`, `user`, `assistant` order. The user message contains one bounded repository-evidence view; the assistant message is the canonical report. |
| `source_url` | Source URL named in the prompt and validated against its source type. |
| `source_type` | One of `aigcodeset`, `synthetic_repository`, `artifact`, or `real_repository`. |
| `source_kind` | `snippet` or `repository`, describing the input shape. |
| `source_groups` | Non-empty lineage identifiers used to prevent related rows crossing splits. |
| `split` | `train`, `validation`, or `test` in a combined candidate. |

The shared [report contract](../../../src/contracts/README.md) governs the conversation and citations. Citations in a target must appear in the evidence view included in that same row.

The assistant message is the whole report the prompt asks for, including the two
`- Why this score:` lines. They name the sign that sets each number and the direction that
would move it, because a target of three numbers and one sentence has nothing to reason
with: the model can fit the score arithmetic from the measurements printed in the prompt and
reach a near-zero loss without ever producing a justification. The parser does not require
the lines, so a stored report without them stays readable.

## Lineage and split isolation

Rows are connected when they share a canonical repository URL, rendered prompt, or source-group identifier. Source-group identifiers also record relationships such as content hashes, upstream problem groups, and synthetic/artifact ancestry. The splitter assigns each connected component to one partition; it does not split a repository into separate partitions just to meet a row quota.

Before splitting, prompts with conflicting targets are quarantined with their connected source component. Safe duplicate prompts are merged while retaining their provenance. Final verification checks source separation, required source families, quotas, and target distribution.

## Scores and labels

| Target type | Default range | Interpretation |
| --- | ---: | --- |
| Human-labeled teaching example | 15–25 | A generated style target below the 40% flag threshold. |
| AI-labeled teaching example | 45–85 | A generated style target above the threshold. |
| Explicit-artifact example | 88–99 | A strong target used for declared artifacts. |
| Verified repository row | measured share | The percentage a harvest attributed, with the band midpoint only when no measurement was recorded. |

An example's distance from an undecided reading, not its raw position, sets the number inside its band: a position at the midpoint sits on the band edge nearest the 40% boundary and an extreme position reaches the far edge. Mapping the position onto the band directly wasted most of it, because a supervised AI example had to reach 0.6 and the reachable range therefore began at 69 — nearly every repository row targeted the same high score, and the served judge returned that one number for every project.

The configurable boundary band defaults to 25–60 so examples can represent both sides of the 40% decision threshold. These are heuristic teaching targets, not human-verified authorship labels. A model agreeing with them does not prove real-world accuracy.

Documentation can never carry a target. `AI_LIKE_FEATURES` and `HUMAN_LIKE_FEATURES` are saturation scales, not weights; what a target actually rests on is the measured separation `feature_weights` turns into shares, and the three README features together are capped at `MAX_DOCUMENTATION_WEIGHT_SHARE` of one. The cap is deliberate: the recorded build log measured them carrying 66% of the weight for the repository rows, so those targets justified a score with "the README has 7 heading(s) and 2 bullet(s)" while the code that decided the score went unmentioned. Code-derived features take the capped remainder, `evidence_phrase` heads a target with a code measurement whenever one is readable, and a documentation feature leads only when no code measurement in the example is readable at all.

No target may quote the project's description or its tooling, and none may cite formatting. The quote selector reads only files `carries_authorship_evidence` accepts, so a README heading, a `package.json` field, or a `.gitignore` line is never the line a target cites, and a batch showing nothing but the description yields no target at all. Indentation and trailing whitespace are not features either: the projection inverts the human-like map, so a project indented with spaces read as fully machine-like for its zero tab count, the same inversion carried 21% of a repository target in the recorded build log, and the served judge answered "0 line(s) are tab indented" for ordinary projects. Neither measurement is supervised or shown to the judge now.

Real-repository rows do not go through that function at all. Their target is the percentage a
verified label measured, and the builder refuses to produce them without a manifest: a
repository measured at 4% agent-attributed lines teaches 4, one measured at 96% teaches 96,
and a repository that cannot supply an auditable record is left out rather than scored on
style. That is deliberate. The repository-shaped style weights were fitted on synthetic
repositories and labeled snippets, and on real code they did not separate declared
AI-generated repositories from ordinary human projects: over the 52-repository pool in
`data/repos.fresh.txt`, human projects sat at a mean position of 0.71 and AI-declared ones at
0.62, so most of both groups landed in the AI band. Measured on the served model, twenty
repositories that ended before 2023 came back at a mean of 82.3 against a detector reading
of 16.4, which is what supervision from that position produced.

## Verified repository labels (fail closed)

Repository authorship has to come from an auditable record, not from the detector's style
reading. When the builder is given a JSONL manifest with
`uv run src/cli.py data -r data/candidate_repos.txt --repository-labels data/repository_labels.jsonl`, only repositories
whose entries pass the fail-closed standard are read, and each accepted label sets the
row's target directly (see `labels.py`). The target is the share the entry measured —
`attributed_lines / examined_lines`, or files when no line counts were recorded — a
verified human-class repository teaches 0, and the band midpoint survives only for an `ai`
entry that recorded a class and no measurement. A repository without an accepted label is
excluded, and `--repos` without a manifest stops the build instead of falling back to the
detector's style reading.

Each entry needs `url`, `label` (`ai` or `human`), `evidence`, `audited_by`, and
`audited_at`. An `ai` label needs an agent-attributed pull request with its merged commit,
or a recorded generation trace. A `human` label needs a controlled collection that
prohibited AI use and recorded its protocol. A README that declares AI authorship is a
claim, not a record, so it is rejected, and `require_usable` stops a build whose manifest
covers only one class.

### Where an accepted manifest comes from

`uv run src/cli.py harvest` builds it. The harvester searches merged pull requests by
registered agent app, resolves the most recent commit that touched each file the reader would
read, and writes the cohort record beside the label manifest those commits support. Each `ai`
entry carries `attributed_files`, `examined_files`, `attributed_lines`, and `examined_lines`,
so a label always shows how much of the repository was measured before it was trusted. The
per-file rule, the verified agent identities it matches, and what it cannot establish are in
[harvest/README.md](harvest/README.md); the measurement that motivated it is in the
[repository AI percentage design](../../../docs/superpowers/specs/2026-10-12-repository-ai-percentage-design.md).

### Building without repository rows

`HACKGRADER_SNIPPET_ONLY=1` produces a candidate with no `real_repository` rows at all, so
no split quota requires one. Use it when you want a model whose style and quality reading
rests only on families whose targets are already defensible — AIGCodeSet labels, synthetic
projects, and explicit artifacts — rather than shipping a repository head trained against
unverified targets. The mode is opt-in and off by default; every other gate is unchanged.

```bash
HACKGRADER_SNIPPET_ONLY=1 uv run src/cli.py data -o dataset.snippet.candidate.jsonl
HACKGRADER_SNIPPET_ONLY=1 uv run src/cli.py train --dataset dataset.snippet.candidate.jsonl --preflight-only
```

Three things it deliberately costs. The model never sees the multi-file dump format the
served grader actually reads, so repository scoring degrades rather than becomes correct.
`--repos` becomes a contradiction and is refused instead of quietly ignored, so a build
cannot end up half in and half out of the mode. And the setting must be present for both
the `data` and the `train` command: it is read at process startup, so a training run
without it re-imposes the two-repositories-per-split floor on a dataset that cannot meet
it.

Measuring the 52-repository `data/repos.fresh.txt` pool against this standard accepts nothing: 8
repositories only declare AI authorship in their own files, and 44 carry no authorship
evidence at all. Treat those repository targets as unverified until audited records exist.

## Common settings

Settings are read when the process starts. Defaults are defined in [`config.py`](config.py); training split requirements are in [`../learning/config.py`](../learning/config.py).

| Environment variable | Default | Effect |
| --- | ---: | --- |
| `HACKGRADER_AIG_ROWS` | `2500` | Requested AIGCodeSet rows per label. |
| `HACKGRADER_SNIPPET_CHARS` | `1500` | Maximum retained code characters per labeled sample. |
| `HACKGRADER_SYNTHETIC_REPOS` | `256` | Requested synthetic project count. |
| `HACKGRADER_SYNTHETIC_FILES` | `2` | Files per synthetic project. |
| `HACKGRADER_ARTIFACT_ROWS` | `48` | Requested explicit-artifact example count. |
| `HACKGRADER_REPO_REPEAT` | `1` | Training repetitions for real-repository rows; does not change validation/test rows. |
| `HACKGRADER_TARGET_CAP` | `1000000` | Maximum retained rows per generated target score. |
| `HACKGRADER_MIN_REAL_REPOSITORIES_PER_SPLIT` | `2` | Independent repositories required in each source-disjoint split. |
| `HACKGRADER_SNIPPET_ONLY` | `0` | Build with no real-repository rows at all, dropping that family from every quota. A positive repository floor alongside it is refused. |
| `HACKGRADER_MIN_SCORE_BAND_ROWS` | `10` | Minimum prompts for configured score-band quotas. |
| `HACKGRADER_REPO_CACHE` / `HACKGRADER_REPO_CACHE_HOURS` | `.cache/repos` / `24` | Cache directory and cache freshness period. |
| `HACKGRADER_HARVEST_COHORT` | `data/cohort.jsonl` | Cohort file the harvest appends to and resumes from. |
| `HACKGRADER_HARVEST_BUDGET` | `5000` | Metered GitHub requests one harvest run may spend. |
| `HACKGRADER_HARVEST_MAX_FILES` | `40` | Files attributed per harvested repository. |
| `HACKGRADER_HARVEST_TOUCH_WINDOW` | `5` | Commits examined per file, which decides the human class. |
| `HACKGRADER_HARVEST_HUMAN_BEFORE` | `2023-01-01` | Date every commit of a human-class repository must predate. |
| `HACKGRADER_HARVEST_PAUSE` | `0` | Seconds before each metered request, for anonymous rate limits. |
| `HACKGRADER_MAX_LEN` | `3072` | Maximum serialized chat-token length. |
| `HACKGRADER_HUMAN_LABEL_BAND` | `15,25` | Target range for human-labeled teaching examples. Distance from the midpoint spreads it: the human-most examples sit 15 points below the cutoff. |
| `HACKGRADER_AI_LABEL_BAND` | `45,85` | Target range for AI-labeled teaching examples. Distance from the midpoint spreads it: a supervised example reaches 53 to 85. |
| `HACKGRADER_ARTIFACT_BAND` | `88,99` | Target range for explicit artifacts. |
| `HACKGRADER_BOUNDARY_BAND` | `25,60` | Range spanning the 40% decision threshold. |

The fresh-repository builder flags are opt-in. They add final-test minimums of 50 real-repository prompts and 10 per score class; they do not bypass any other quota. The fresh-repository training gate filters the final test to real-repository rows whose canonical URLs are absent from train and validation. See [training preflight](../learning/preflight/README.md).

## Module map

| File | Responsibility |
| --- | --- |
| `__init__.py` | Adds the shared `contracts` source path so the package can import it. |
| `config.py` | Configures candidate sources, split quotas, and score targets. |
| `examples.py` | Loads and wraps labeled AIGCodeSet samples with stable source lineage. |
| `sources.py` | Loads cached real-repository evidence for candidate generation. |
| `synthetic.py` | Builds repository-shaped synthetic and explicit AI-artifact examples. |
| `features.py` | Measures model-visible features and assigns evidence-weighted score targets. |
| `evidence.py` | Attaches source-grounded evidence to target examples. |
| `labels.py` | Fail-closed repository authorship labels and their provenance score bands. |
| `rows.py` | Summarizes, filters, and deduplicates finalized candidate rows. |
| `rendering.py` | Renders evidence examples into the shared live grading conversation format. |
| `validation.py` | Validates chat rows against provenance and the live report contract. |
| `verification.py` | Verifies split lineage, message contracts, and target integrity. |
| `publishing.py` | Publishes candidate JSONL files only after proving they fit the trainer window. |
| `harvest/` | Harvests the agent-attributed repository cohort and the fail-closed labels it supports. |

## Publication and safety

Run the audited build from `model/` with `./scripts/build_candidate_from_harvest.sh`; it harvests verified labels, derives `data/candidate_repos.txt` from the manifest, and then runs the builder. The default combined output is `data/dataset.candidate.jsonl`. Publication follows format, citation, tokenizer-length, split-size, lineage, and quota checks; an unsuccessful build does not publish an incomplete candidate. Output paths must use `.candidate.jsonl` and stay within `model/`; active train, validation, and test files are protected.

This package describes the dataset interface; it does not establish that source repositories are novel to a pretrained model or that heuristic targets are ground truth. Review candidate evidence and split coverage before training.
