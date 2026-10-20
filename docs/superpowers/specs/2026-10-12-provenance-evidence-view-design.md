# Provenance-carrying evidence view — design spec

Date: 2026
Status: SUPERSEDED — see `2026-10-12-repository-ai-percentage-design.md`
Author: Quentin Bordelon
License: MIT

The diagnostic tools and logs this design cites — `audit_repository_labels.py`,
`measure_repository_separation.py`, `separation_baseline.log`, and
`trailer_prevalence.log` — were removed from the workspace in the 2026-10-12 cleanup, after
the direction below was superseded. The measurements they recorded are quoted inline here, and the model-side learning and training-data suites named in §10 were removed in the 2026-10-13 cleanup, so the paths in that table record what was designed rather than files that exist today.

## 1. Problem

Every repository row is supervised from an audited label in `model/src/data/labels.py`, but the
prompt the model actually sees carries almost none of the evidence that label rests on. The
baseline in `separation_baseline.log` measures the
consequence:

| Observation | human | ai_declared |
| --- | --- | --- |
| Views carrying an explicit generation artifact | 0 / 652 (0.0%) | 9 / 132 (6.8%) |
| Mean detector `ai%` | 11.2 | 14.1 |
| Views surviving the label gate | 7 / 652 (1.1%) | 74 / 132 (56.1%) |
| Standardized position gap | +0.418 sd, wrong sign | |

The signal itself is not the problem. Provenance discriminates **perfectly** where it appears — zero
false positives across 652 human views — and covers 6.8% of AI views. That 6.8% is an accident of
file content: the only way a generator is named today is if some file body the reader happened to
reproduce contains a banner. No stage of the pipeline requests commit or pull-request metadata, so
the view cannot carry what the label is built from. Coverage is the defect, and the label is not
learnable at inference time until the view carries the evidence.

### Measured on 2026-10-12

Before writing any code, the pool's commit histories were scanned directly (one
`GET /commits?per_page=20` per repository, 52/52 readable, authenticated; full output in
`trailer_prevalence.log`):

| | repos | with a strict trailer | views | views lost | share |
| --- | --- | --- | --- | --- | --- |
| human section | 43 | 11 (25.6%) | 652 | 171 | 26.2% |
| AI-declared | 9 | 3 (33.3%) | 132 | 44 | 33.3% |

The decisive number is the human one: a trailer appears in **11 of 43 ordinary repositories**,
so the evidence is not rare in the class it would have to leave alone. The agents are identical
on both sides — `claude` in 9 human repositories and 2 AI-declared ones, `copilot` in 4 and 1 —
and ordinary projects such as `psf/black`, `rails/rails`, `redis/redis`, `nlohmann/json` and
`catchorg/Catch2` carry `Co-authored-by: Claude` in their last 20 commits.

The 9-repository AI-declared class is too small for its own 33.3% to be estimated precisely, and
it does not need to be: 6 of the 9 show no trailer at all in that window, and all 9 already carry
an in-file banner in some view, so trailers cannot carry that class either.

Two consequences follow, and they drive the revised decisions below. Treating a trailer as
explicit-artifact evidence would push 171 human views toward the artifact band while adding
only 44 AI views, and would teach a rule the population contradicts. And the fail-closed
drop of trailer-carrying human repositories costs 26.2% of that class, so a rule that drops
them is affordable but must be deliberate.

## 2. Decisions

Approved on 2026-10-12:

1. **Repository-level provenance.** The view carries one provenance block per repository view,
   derived from the head commit and recent commit messages. Per-file last-commit attribution is out
   of scope: it costs one API call per included file (up to 120), which would exhaust an
   unauthenticated budget and roughly triple collection time.
2. **Degrade legibly, never silently.** When the API cannot be read, the view states that provenance
   was not retrieved and why. Silence is indistinguishable from a clean history, so it is not an
   option.

Because granularity is repository-level, the block must never be phrased as if it identified a single
file. Section 5 makes that a testable requirement.

## 3. Where the block is produced

The block is emitted by the collector, `src/repositories/collection.py`, and by nothing else. Both
the serving and the training path already consume that one function:

- Serving: `src/grading/runtime.py` → `repositories.reader.read_repository` → `collect_repository`.
- Training: `model/src/data/sources.py::repo_fetcher` → the same `read_repository`, cached under
  `REPO_CACHE_DIR` for `REPO_CACHE_HOURS`.

Emitting inside the collector is what makes contract parity structural rather than aspirational: the
training path never re-fetches and never rewrites a header, so it cannot drift from what the served
model sees. Tempting alternatives that do **not** work, and why:

- Adding the block in `model/src/data/rendering.py` would put it in training only. The served prompt
  would keep the defect.
- Adding it in `src/grading/runtime.py` would put it in serving only, so the target would be
  learnable nowhere.
- Deriving it from the cached dump in the probe would measure a view the model never sees.

The block is **optional**. A dump without it renders exactly as today, so existing fixtures, cached
dumps, and snippet rows are unaffected.

## 4. Block contents

Three lines, inserted directly after `DEFAULT BRANCH:` and before `FILES FOUND:`:

```text
PROVENANCE: github-api head commit 0f1e2d3c4b5a69788796a5b4c3d2e1f001234567 dated 2026-09-30 by alex
PROVENANCE GENERATION TRAILER: co-authored-by: copilot (3 of 20 recent commits)
PROVENANCE HEAD PULL REQUEST: #123 merged
```

Sources, and their cost:

| Line | Source | Calls |
| --- | --- | --- |
| Head commit SHA, author date, author login | `GET /repos/{o}/{r}/commits?per_page=20` | 1 |
| Generation trailer and its frequency | the same 20 commit messages | 0 |
| Head pull request state | `GET /repos/{o}/{r}/commits/{sha}/pulls` | 1 (optional) |

One call buys both the head identity and a 20-commit trailer scan, so trailer coverage comes from the
commit history rather than from the head commit alone — a merge commit at the head carries no trailer,
while the commits beneath it do. The pull-request line is best-effort: when that call is not made or
fails, the line reads `unknown`.

Unavailable rendering:

```text
PROVENANCE: unavailable (HTTP 403)
```

Every branch is bounded and enumerated:

- **Author login** is echoed only when it matches `^[A-Za-z0-9-]{1,39}$`; otherwise `unknown`.
- **Trailer value** comes from a controlled vocabulary, never from raw commit text. The complete
  set is: `co-authored-by: copilot`, `co-authored-by: claude`, `co-authored-by: devin`,
  `co-authored-by: cursor`, `co-authored-by: codex`, `co-authored-by: gemini`,
  `generated with copilot`, `generated with claude`, `generated with devin`, `generated with cursor`,
  `generated with codex`, `generated with gemini`, `as an ai language model`,
  `other recognized agent trailer`, and `none found`. The frequency suffix is
  `(n of 20 recent commits)`.
- **No free-form commit message text is ever included.** This is a correctness requirement, not a
  style preference. `detection.metrics` counts emoji, `TODO`/`FIXME` notes, debug prints, and
  local paths over the **whole** view text, and `metrics["lines"]` counts every non-blank line
  outside a `--- FILE:` block. Arbitrary commit text would therefore inject human-trace and
  emoji counts straight into the features that the separation measurement depends on, letting an
  unrelated commit message move the target.
- The block is at most 3 lines and 320 characters, which bounds per-batch header overhead.

The wording above is chosen for readability, and it deliberately reads like the artifact it
describes. It cannot be made safe by phrasing: `co-authored-by: copilot` and
`generated with claude` both match the **existing** banner alternation in `AI_SIGNALS`, which is scanned
over the whole view with `allow_documentation=True`. What keeps the block out of the authorship
measurement is an explicit exclusion in section 5, not a euphemism here.

## 5. Interaction with the detector and the measurements

The block is evidence the model may cite. It must **not** become an authorship signal for the
detector.

**No artifact weight for the provenance block.** An earlier draft of this spec added a
`PROVENANCE GENERATION TRAILER` entry to `AI_SIGNALS` at `EXPLICIT_ARTIFACT_WEIGHT`. The measurement in
section 1 rules that out. Weighted that way, the trailer contributes to `ai_score` on 171 human views
against 44 AI views, and under `provenance_target` it does one of two harmful things: it drops a human
repository whose history happens to carry a `Co-authored-by: Claude` line, discarding a real
counterexample, or — where the row survives — it teaches that a trailer implies the 88-99 artifact
band. A student project with a single Copilot-suggested commit would then score top-band, which is the
false-positive failure this project already paid for once. The tournament is not evidence: the same
two agents appear at a similar rate on both sides of this pool.

The rule that follows: `has_explicit_artifact` is computed from **file bodies only**, exactly as today,
so the trailer neither raises `ai_score` nor drops a row. The block is carried, is quotable, and is
something the model must reason about in context rather than a bound it can trip. A human-supervised
row with a trailer in its provenance is therefore a *useful* example: it shows the model a repository
whose code reads human while its history names a tool.

**The artifact scan must ignore provenance lines.** `ai_findings` skips any line whose stripped form
starts with `PROVENANCE`, in the same way header lines are already excluded from citations. Without
this, the natural trailer wording collides with the existing banner pattern and the block becomes an
artifact by accident — reintroducing exactly the false rule this section rejects, and doing it
silently. The test is an equality: for one view, `ai_score`, `quality_score`, `has_explicit_artifact`
and every `metrics` field except `lines` and `chars` must be identical with and without the block. Any
header-prefix exclusion is also the reason the block must never be reworded to describe *files*.

**Palette mirroring still applies to in-file banners.** A gap this surfaced: the existing banner
alternation names `chatgpt`, `gpt`, `copilot`, `claude`,
`cursor`, `codex` and `gemini` but **not `devin`**, and it recognizes a `co-authored-by` line for
`copilot` alone, while `model/src/data/labels.py` records the contributing agent without constraining
its name (its documented example is `devin`) and `audit_repository_labels.py` scans repository files
for a devin trailer. An audited `ai` repository contributed by devin could be labelled but never
supported by its own view, and a file *body* carrying `Co-authored-by: Claude <noreply@anthropic.com>`
would be missed. One agent list, shared by the banner alternation and the trailer vocabulary, pinned
by a test so they cannot diverge — with the artifact weight confined to the banner path.

**Invariance requirement.** Adding the block must not disturb any measurement taken over file bodies.
For the same file bodies, with and without the block, these must be byte-identical: `source_files`,
`source_lines`, `code_lines`, `comment_lines`, `spacing_sites`, `spacing_slips`, `tab_lines`,
`trailing_lines`, `functions`, `docstrings`, `restated_docstrings`, `typed_functions`, `readme_chars`,
`readme_headings`, `readme_bullets`, `readme_first_person`, and every `COUNTED_PATTERNS` field. Only
`lines` and `chars` may move, and only by exactly the block's line and character counts. This holds
because `_source_blocks` reads only inside `--- FILE:` blocks; it is a requirement because the
block sits in the header, and a regression here would silently shift every repository target.

**Citation integrity.** `verify_citations` accepts a quote that appears anywhere in the view, so a
model citing `PROVENANCE GENERATION TRAILER: co-authored-by: copilot` correctly passes. The converse
must not regress: a citation naming a file absent from the view must still fail, and the block must
not introduce file-like tokens (`FILELIKE_TOKEN` matches `something.ext`) that would let an
otherwise-unsupported citation pass. Because the exact string `PROVENANCE GENERATION TRAILER:
co-authored-by: copilot` must be quotable, the requirement is stated as an invariant instead: the
block contains no token matching `FILELIKE_TOKEN`, which is why the author login is restricted to
alphanumerics and hyphens, no email address is echoed, and the SHA carries no dot-suffixed suffix. A
test asserts that a citation naming a file absent from the view still fails on a view carrying the
block.

## 6. Where it appears in the prompts

**Serving.** `collect_repository` renders the block into the header, which lands inside
`===== BEGIN REPOSITORY CONTENT =====` in `contracts.rendering.build_user_message`. When a repository
is split across batches, `contracts.repository_dump._base_header` rebuilds the header for **every**
batch, so the block must be carried through it exactly as `REPOSITORY:` and `DEFAULT BRANCH:` are
today. It follows that `RepositoryDump` gains a `provenance` field holding the block's lines, and
`_base_header` re-emits them verbatim; `trim_dump`'s `_compact_header` shortens path lists and must
leave provenance lines untouched.

**Training.** `model/src/data/rendering.py::_finalize_core` calls `report.split_dump` on the cached
dump and therefore reaches the same `_base_header`. Parity is expected to be automatic; a test proves
it by rendering one dump through both paths at the same budget and comparing the provenance lines
byte for byte.

**Budget.** The block adds up to 320 characters to every batch header. `split_dump` raises when
`budget - len(header) - len(file_header) <= 0`, and `_finalize_core` retries downward toward a 256
character floor before raising `Cannot fit <source_type> source`. The new floor must be recomputed and
pinned by a test, so a future budget change fails loudly in the builder rather than in production.

**Rubric.** `SYSTEM_PROMPT` in `src/contracts/prompts.py` gains one sentence stating that
`PROVENANCE` lines describe the repository's commit history, apply to no individual file, and may be
cited as repository-level evidence. Without it a repository-level trailer invites a per-file claim
that the citation verifier cannot catch, because the trailer genuinely is in the view. The system
prompt is shared by both paths, so this stays consistent by construction.

## 7. Acceptance criteria for the gate

Run after the change, from `model/`, with `GITHUB_TOKEN` set:

```bash
uv run python measure_repository_separation.py --dataset dataset.candidate.jsonl
```

Two calls per repository is up to 104 calls for the 52-repository pool, which is why a token is
required: the unauthenticated budget of 60 requests per hour cannot cover the pool, and the run would
degrade to `unavailable` lines for the remainder. **Before running, invalidate the pool cache.** A
cached dump younger than `REPO_CACHE_HOURS` is reused as-is and would carry no provenance, which
would read as a coverage failure that is purely a tooling artifact. Run the probe with
`HACKGRADER_REPO_CACHE_HOURS=0`, which makes every cached dump non-fresh and forces a live read
without deleting anything.

Every **binding** row must clear before any training run starts. The other two are diagnostics, and
which is which matters more than it looks:

| # | Criterion | Binding? | Required | Today |
| --- | --- | --- | --- | --- |
| 1 | Provenance block present | yes | ≥ 95% of views per class | 0%, field does not exist |
| 2 | Among present blocks, `unavailable` | yes | ≤ 5% per class | n/a |
| 5 | Verified-path retention (`_finalize_core`'s path) | yes | ≥ 60% per class, drop reasons broken down | 100% on `provenance_target`, 1.1% on the style gate |
| 6 | Measurement invariance (section 5) | yes | exact | untested |
| 7 | Contract parity (section 6) | yes | byte-identical | untested |
| 8 | Citation integrity (section 5) | yes | exact | untested |
| 3 | Standardized position separation, AI higher | no | ≥ 25 points, correct sign | +0.418 sd, wrong sign |
| 4 | Detector `ai%` gap | no | AI ≥ human, gap ≥ 5 points, zero contribution from provenance lines | 11.2 vs 14.1, inverted |

Rows 3 and 4 cannot be binding for a manifest-driven build, and requiring them would block training
forever. Both measure the *style* reading, while `_finalize_core` scores every repository row through
`provenance_target`: the verified label, not the style position, decides the band, and section 5 makes
the block measurement-invariant, so neither number can move as a result of this change. They stay in
the report because the same probe measures unlabeled rows and the pool's own declarations, and there
the style reading is the right test — and because a verified label whose repository the detector reads
as machine-like is worth seeing. It must not be a precondition.

This also corrects the rule carried over from the earlier failure: it is any *binding* failure that
stops the run, not any failure.

**An instrument gap to close first.** The probe's reported "label gate" is
`features.label_target`, the *style* gate, while `_finalize_core` scores a verified row through
`provenance_target`, which never consults the style position and drops only on a file-body artifact.
The two are different gates, so the current 1.1% human retention describes a path the builder does not
use for verified rows. Criterion 5 must be measured on `provenance_target`, and the style gate
reported alongside it for contrast rather than in its place.

**Not evaluable is not a pass.** A run without `GITHUB_TOKEN`, or one whose cache was not refreshed,
reports NOT EVALUABLE for rows 1-2 and must not be recorded as clearing the gate. Criterion 5 must
report *why* each dropped view was dropped — a file-body banner versus a provenance trailer — because
the measured 26.2% trailer incidence means a rule that drops trailer rows would remove a quarter of
the human class, and that must be a visible, chosen cost rather than a silent one.

Criteria 1-2 are what "coverage" means, concretely, and they are the reason the earlier failure becomes
diagnosable. Criterion 5 decides whether the human class survives the fail-closed rule, and section 1
already answers it: with the trailer carrying no artifact weight, provenance drops nothing in this
pool, so retention is the 100% `provenance_target` reports today. The open question is therefore not
retention but whether the model learns to weigh a trailer correctly — which rows 6-8 and the human
counterexamples in section 5 exist to protect.

## 8. Risks

- **Pool control.** Measured, not hypothetical: 11 of the 43 human-section repositories carry a
  generator trailer, and the agents are the same ones the AI class names. This is why the trailer
  carries no artifact weight (section 5). The residual risk is the opposite one — a model that sees a
  trailer and inflates a score anyway — which is why a human-supervised row *with* a trailer is
  deliberately kept in the training data rather than dropped.
- **Header overhead.** Three lines per batch is a real cost against `PROMPT_TOKEN_BUDGET`. The 320
  character bound keeps it predictable, and section 6's floor test keeps a regression visible.
- **Stale cache.** The highest-probability way to misread this change as a failure is reusing
  pre-change cached dumps. Section 7 makes invalidation an explicit step.
- **Rate limit.** Criterion 2 cannot reach 95% without a token. This is a hard prerequisite of
  measurement, not a nice-to-have.

## 9. Out of scope

Per-file commit attribution. Any change to the label standard or the score bands. Any change to the
detector's non-explicit weights. Fetching repository history beyond 20 commit messages. Collecting the
audited cohort — that stays blocked until this gate clears.

## 10. Deliverables

| File | Change |
| --- | --- |
| [`src/repositories/collection.py`](../../../src/repositories/collection.py) | Fetch head commit and trailer scan, build the block, pass it to `_render_dump` |
| [`src/repositories/settings.py`](../../../src/repositories/settings.py) | Trailer vocabulary, commit-scan size, block character bound |
| [`src/contracts/prompts.py`](../../../src/contracts/prompts.py) | Header constants and regexes, one rubric sentence |
| [`src/contracts/repository_dump.py`](../../../src/contracts/repository_dump.py) | Parse the block, carry it on `RepositoryDump`, re-emit it in `_base_header` |
| [`src/detection/signals.py`](../../../src/detection/signals.py) | `PROVENANCE` lines excluded from the artifact scan; agent palette mirrored for in-file banners |
| [`model/src/data/evidence.py`](../../../model/src/data/evidence.py) | Treat `PROVENANCE` lines as headers, add the provenance evidence branch |
| `model/measure_repository_separation.py` (no longer in the workspace) | Presence, declaration, and drop-reason metrics; NOT EVALUABLE reporting |
| `tests/training_contract/test_provenance_view.py` (planned only; that suite is no longer in the workspace) | New contract tests (no comments, at most 200 lines, per the source-constraint test) |

Existing fixtures that build a header by hand keep working unchanged, because the block is optional.
