# Repository AI-Code Percentage: Restart Design

Author: Quentin Bordelon
Date: 2026
Status: Draft — awaiting review before implementation
Supersedes: `2026-10-12-provenance-evidence-view-design.md`

## 1. The deliverable, stated as a number

The only output that matters:

```
AI% = AI-attributed source lines in the reported views
      ────────────────────────────────────────────────
      source lines in the reported views
```

Every percentage is reported with **coverage** = reproduced lines ÷ lines the reader found,
so a percentage is never read as if it covered code the reader never saw. Where provenance
exists, `AI%` is a *measured* fraction. Where provenance does not exist (a student repository
with no meaningful commit history), `AI%` is an *estimate of that fraction from code content
alone* and must be labelled as one.

Accuracy therefore has a defined meaning: the estimate's distance from the computed fraction
on repositories where the fraction can be computed independently. Today it has no such meaning;
§2 shows why.

## 2. Why the current system cannot be accurate, by construction

Three independent defects, all in the *target*, not the training loop:

1. **Bands replace the number.** Training targets are band midpoints (human 15–25, AI 45–85,
   artifact 88–99) and the accepted error is MAE ≤ 20 points. Sampling a midpoint as truth
   caps achievable accuracy at roughly half a band width regardless of model quality. A model
   that learned the bands perfectly would still be unable to return `AI% = 63` for a repository
   whose true value is 63.
2. **Repository targets are style-derived.** Repository-level labels were produced by detector
   heuristics over code style. Training a model to predict a heuristic's output reproduces the
   heuristic, including a measured sign inversion (below-median AI repositories scored *above*
   median). The supervision contains no independent information about authorship.
3. **The serving path can overwrite the model with that same heuristic.** In
   [aggregation.py:71](../../../src/contracts/aggregation.py#L71), if any citation is unsupported the
   aggregate score is replaced wholesale by `detector_ai_score`. The output can therefore be
   the uncalibrated style heuristic rather than the trained model, invisibly.

### 2.1 What is already correct: composition computes a fraction

`combine_reports` weights each batch report by its source-line count and returns
`Σ(score_i × lines_i) / Σ(lines_i)`. If each `score_i` is the share of AI-attributed lines in
view *i*, that weighted mean **is** the repository's overall share. The aggregation math is
already the right estimator; it does not need replacing. This is the single largest reason the
restart is cheap: the pipeline is already percentage-shaped from the prompt to the response.

## 3. What is reused, changed, and dropped

**Reused unchanged.** Repository reader and dump format; view-level batching and the batch
budget; the line-weighted aggregation in `src/contracts/aggregation.py`; the QLoRA training
loop, checkpointing, recovery, and preflight gates; the serving model and `Modelfile` path;
the report rendering and citation-verification contract.

**Changed.**

| Area | From | To |
|---|---|---|
| Target value | band midpoint (categorical) | share of AI-attributed lines (continuous) |
| Repository supervision | detector style score | computed attribution from merged agent PRs |
| Gates | band-coverage and threshold gates | MAE and calibration on held-out repositories |
| Aggregation on unsupported citation | replace with detector score | fail the view, or report a flagged estimate; never silent substitution |
| Evaluation | internal separation only | held-out repositories with computed truth |

**Dropped.** Detector-derived repository targets; band-midpoint regression as the accuracy
claim. Detector signals survive as *features and evidence*, never as the label or the output.

## 4. Ground truth: merged agent-authored pull requests (measured)

Verified live against the GitHub API during this design:

- **1,962,950 merged PRs** authored by agent apps: Copilot SWE-agent 1,535,032 · Jules 204,462 ·
  Devin 189,734 · Cursor 29,715 · Codegen 4,007.
- The filter is a real filter, not an ignored parameter: an unknown app slug returns HTTP 422,
  not a full count.
- Hits are genuinely bot-authored (`devin-ai-integration[bot]`, `type=Bot`).
- **File-level attribution is retrievable.** Sampled merged PR `AIprojectDemo/Polydemo#55`:
  `changed_files=1, additions=30, deletions=1`, file `src/plot_coex_3d.py`.
- Quota: core 5,000 requests/hour; search 30 requests/minute. Search returns at most 1,000
  results per query, so the cohort must be partitioned by agent × date range.

A merged agent PR is a signed statement of *which files and how many lines an AI authored*.
It is auditable, reproducible, and independent of code style — the property the current
supervision lacks.

### 4.1 Attribution rule: label at the granularity we read

The reader reads **files**, not diffs, and the model is asked to score **a view of files**.
So the label must be per file, not per PR:

> A file is **AI-attributed** when its most recent commit touching it was authored by a
> registered agent app. Its AI line count is the number of lines that commit added to it,
> bounded by the file's current length.

Obtained from `GET /repos/{owner}/{repo}/commits?path={file}` (one request per file, core quota),
which distinguishes files an agent wrote from files an agent merely touched, and from files a
human later rewrote. A repository's truth is then exactly §1's fraction, computed per file and
summed — the same arithmetic the aggregation already performs. Repository-level PR-diff
attribution is rejected as the primary rule: it over-counts code an agent added and a human
later replaced, and under-counts files an agent edited without changing their length.

### 4.2 Classes from one source

Because the cohort is repositories, not snippets, both classes come from the same source:

- **AI-attributed files** — files whose last touching commit is agent-authored.
- **Human files** — files in the *same repositories* never touched by any agent commit (the
  strongest control: matched language, project, and era), plus a documented temporal cutoff
  (pre-LLM code) as a secondary class.

### 4.3 Cohort requirements

- Whole repositories, held out **by repository**, never by file.
- At least 60 repositories spanning ≥3 languages, each with both classes present.
- Per-repository truth recorded with the commit SHAs and the rule version used to compute it.
- A manifest records the exact queries, partitions, timestamps, and rate-limit budget spent,
  so the cohort is reproducible rather than asserted.

## 5. What the model learns

**Training signal 1 — the cohort (§4).** Continuous per-view targets with computed truth.
This teaches the mapping from code to share-of-AI-lines, calibrated against measurement.

**Training signal 2 — content-labelled code (AIGCodeSet, already wired in).** This exists to
solve the failure that sinks a provenance-only plan: a student hackathon repository has no
usable commit history, so a score must come from the code itself. These rows carry
LLM-versus-human labels on code content, which is exactly the signal that generalises to
provenance-free input. They dominate the training mix; the cohort supplies evaluation and
calibration.

**The trap to avoid.** Training on file-level labels derived from commit history and then
serving provenance-free repositories invites the model to learn *provenance ⇒ AI*. The guard
already exists in the prompt, verified at
[prompts.py:101](../../../src/contracts/prompts.py#L101): the reader strips commit history, and the
system prompt explicitly forbids inferring authorship from the URL, owner, repository name,
or file names. Every training row must stay renderable from code content alone, and the risk
is verified empirically by evaluating the cohort with provenance removed. The one authored
signal the prompt does permit — a self-naming `generated by` banner or co-author trailer
([prompts.py:97](../../../src/contracts/prompts.py#L97)) — is code content, so it survives this test.

**Output contract unchanged.** The assistant message keeps the rendered report format
(percentage + evidence quotes + summary), so `validate_row`, citation verification, and
`parse_report` continue to work. Only the *distribution of the number* changes: continuous,
not banded.

## 6. Evaluation: the acceptance test that does not exist yet

For each held-out repository, compare the model's `AI%` against the computed fraction:

- **MAE** in percentage points, per repository and per view, with coverage reported alongside.
- **Calibration**: bin predicted values and record the mean true fraction per bin; a well
  calibrated model's bins track the diagonal. This catches band-shaped collapse, where all
  predictions pile on a few round numbers.
- **Baselines, reported together.** (a) the existing detector's score, (b) always predicting
  the cohort mean, (c) the band-midpoint behavior the current model exhibits. The claim "this
  is accurate" is only admissible if the new model beats these on held-out repositories.
- **Provenance-free slice.** The same repositories evaluated from code content only, since
  that is the user's real input. Reported separately with its stated limitation (§1).

**Proposed gates** (calibrated once the baseline in §6 (a)–(c) is measured, before training).
The MAE gate already exists as a single environment-tunable constant —
`HOLDOUT_MAE_LIMIT`, default `20`, enforced in
[threshold_violations.py:20](../../../model/src/learning/holdout/threshold_violations.py#L20) — so
tightening it to 10 is a recorded configuration change rather than new machinery. The missing
piece is calibration, which is not measured anywhere today.

- Repository-level MAE ≤ 10 points on the held-out cohort, versus the measured baseline.
- No calibration bin off the diagonal by more than 15 points.
- Coverage reported for 100% of outputs; no output without a coverage figure.
- Zero silent detector substitutions.

### 6.1 First measured baseline (2026-10-12)

The harvester and this measurement ran end to end on live data: three repositories, both
classes present in two of them, attributing 12 agent-authored files against 10
human-last-touch files.

| Scorer | MAE, 3 repositories | MAE where truth is ≥50% visible (2) |
|---|---|---|
| detector (style heuristic) | 20.58 points | 13.81 points |
| always_mean constant | 2.36 points | 1.31 points |

The detector sits roughly ten times further from the computed fraction than a constant
does. On the smallest repository the deployed model was measured through the real serving
path: computed truth 7.5%, detector 31%, **model 40.0%** — an error of 32.5 points, and a
value landing exactly on the 40% flag threshold rather than a calibrated fraction. That is
the band collapse of §2, measured rather than argued.

Two limitations are recorded with these numbers. The model figure is a single repository,
because the serving path takes minutes per repository and did not complete three inside a
foreground run; the full model baseline is a background job. And the reader exposed 79%,
100%, and 15% of the attributed lines respectively, which is why every figure is also
reported on the majority-visible subset instead of only pooled.

## 7. Ordered plan

1. **Harvest the cohort** — a harvester that walks agent PRs, resolves per-file attribution
   (§4.1), fetches file contents through the existing reader, and writes the manifest with
   recorded provenance and budget. Runnable and cacheable; safe to restart.
2. **Measure the baseline** — run the existing detector and current model over the held-out
   cohort and record MAE, calibration, and the three baselines. This number is what the
   restart must beat and it should exist before training starts. *Started: the detector and
   one model measurement are recorded in §6.1; calibration is not yet measured and the
   model baseline needs a background run over the full cohort.*
3. **Retarget the builder** — continuous per-view percentage targets from the manifest,
   content-labelled rows dominating, no provenance in any prompt.
4. **Train** on the reused QLoRA loop with the new targets.
5. **Evaluate** per §6; iterate on the data mix, not on the bands.
6. **Serve and report** with coverage, replacing the detector-override path in aggregation.

## 8. Risks and their handling

- **Attribution noise** (force-pushes, squashes, later human rewrites): mitigated by the
  last-touch rule (§4.1) and by recording rule version and SHAs; noise is *measured* as
  disagreement between the two attribution rules and reported, not hidden.
- **Cohort is not student code.** Agents produce structured PRs; students paste snippets.
  Mitigated by the content-labelled training mix and by evaluating the provenance-free slice.
- **Harvest cost.** Core quota allows ~5,000 file-history requests/hour; the cohort is sized
  to what is genuinely needed (≥60 repositories) rather than maximised. Cache aggressively.
- **Label leakage through provenance.** Handled in §5 by construction and verified by the
  provenance-free evaluation.
- **Scope creep.** The pipeline already works; this plan changes targets, supervision, and
  evaluation, and must not become a rewrite of the reader, loop, or serving path.

## 9. Definition of done

- Cohort harvested with recorded provenance: ≥60 repositories, ≥3 languages, both classes
  present, manifest reproducible.
- Held-out repositories excluded from training by repository identity.
- Baseline (§7.2) measured and reported before training.
- Model MAE ≤ 10 points on held-out repositories and passes the calibration gate, or the
  shortfall is reported with its cause.
- No output without coverage; no silent detector substitution.
- Tests cover: fraction arithmetic in aggregation, coverage reporting, held-out exclusion by
  repository, and absence of provenance in rendered prompts.
