# Judge-Owned Scores: Removing the Detector's Score Substitutions

Author: Quentin Bordelon
Date: 2026
Status: Implemented on 2026-10-14 (see §10 for the measured effect)
Advances: §3 ("Aggregation on unsupported citation") and §6 ("Zero silent detector substitutions")
of [`2026-10-12-repository-ai-percentage-design.md`](2026-10-12-repository-ai-percentage-design.md)

## 1. Intent

The judge's numbers are the reported numbers.

Every published score in `scores.json` — `ai_score` and `code_score` — comes from the judge's
own report, in every case, with no substitution. The detector keeps running, keeps supplying the
measurement block the judge was trained to read, and keeps every number it computes in the
record beside the judge's. It stops deciding the score.

The immediate motivation is that the detector currently overrules the judge in five places, and
one of them applies to **every** successful repository: the published code-quality score is the
detector's, never the judge's, whenever a detector analysis exists. The judge's quality score is
therefore not published anywhere today.

The deeper motivation is already recorded in the restart design, measured rather than argued:
the detector sits 20.58 points from computed truth on three repositories where a constant
(`always_mean`) sits 2.36 (§6.1 of that document). A heuristic roughly ten times further from
the truth than a constant should not be the value that wins when the trained model and the
heuristic disagree — and under §6's acceptance criteria, "zero silent detector substitutions" is
already a definition-of-done item. This spec implements that item for the serving path.

## 2. The five places the detector decides the score today

| # | Site | Condition | Effect |
| --- | --- | --- | --- |
| 1 | [`reporting.py`](../../../src/grading/reporting.py) `_reported_ai_score` | no parseable judge score | `ai_score` = detector's |
| 2 | same, via `_flag_unsupported` path | any unsupported citation | `ai_score` = detector's |
| 3 | same, via `_judge_above_evidence` | judge ≥25 above a detector measurement at or below the 40 flag threshold, no quoted artifact | `ai_score` = detector's |
| 4 | [`aggregation.py:79`](../../../src/contracts/aggregation.py) `combine_reports` | any unsupported citation in any kept batch | the **aggregate** score is replaced before `Reporting` ever sees it |
| 5 | [`reporting.py`](../../../src/grading/reporting.py) `_annotate` | a detector analysis exists (i.e. almost always) | `code_score` = detector's `quality_score`, unconditionally |

Site 4 is the one the restart design names explicitly; sites 1–3 and 5 are the same defect
reached by different doors. Site 5 is the widest: it needs no failure of any kind.

## 3. Approved decisions

**D1 — The judge decides both scores, always.** `ai_score` is the judge's `ai_score`;
`code_score` is the judge's `code_score`. No path substitutes either.

**D2 — Divergence is a flag, not a substitution.** The ≥25-point judge/detector gap keeps its
`JUDGE_ABOVE_EVIDENCE` flag and its explanatory finding, and quality divergence keeps
`MODEL_DISAGREES`. Neither rewrites the number any more. The finding wording changes to state
both numbers and which one was published.

**D3 — Coverage stays.** `files_shown` / `files_found` / `files_fetched` / `truncated`, the
`LOW_COVERAGE` flag, and the PDF's coverage column are kept. Coverage describes how much of the
repository was shown, which is a fact about the run rather than a judgement about authorship,
and it is the one detector output that does not compete with the judge's number.

**D4 — A repository with no parseable report reports no score.** `ai_score` and `code_score`
stay `null` and the project carries `EVALUATION_FAILED`. The PDF already renders that as `n/a`
with a failure panel; the raw output stays in `batch_reports` and `content`.

**D5 — The evidence panels switch to the judge's own citations.** See §5.1; this is the one
element that goes beyond the four questions asked, and it can be struck without affecting D1–D4.

**D6 — The detector stays in the prompt and in training.** Nothing changes in
[`rendering.py`](../../../src/contracts/rendering.py), the measurement block, or the model
workspace's teaching labels. The judge keeps reading the measurements it was fine-tuned with,
so its input distribution is unchanged and its scores remain comparable to earlier runs.

**D7 — The PDF command re-annotates before rendering, sharing the pipeline's pass.** A stored
scores file records the annotation of the build that wrote it, so rendering it unchanged after
the annotation code changes prints the older reading — which is exactly what happened when this
change shipped: a rerun of `hackgrader-pdf` reproduced the old detector-substituted numbers from
a file the *new* code would no longer produce. `hackgrader-pdf` therefore re-annotates the
loaded projects by default and `--no-refresh` renders the file as stored, which is how an
earlier report is reproduced. The pass is extracted from `Reporting.generate_report()` into
`Reporting.annotate_all()` and called by both, because two copies of an annotation pipeline
drift; it reads only data already inside the input file, never writes to it, and makes no
network or model call.

## 4. What changes, file by file

| File | Change |
| --- | --- |
| [`contracts/aggregation.py`](../../../src/contracts/aggregation.py) | Drop the `detector_ai_score` parameter and the `if unsupported: ai_score = detector_ai_score` substitution. The citation tally and `unsupported_citations` reporting stay; only the overwrite goes. |
| [`grading/evaluation.py`](../../../src/grading/evaluation.py) | `combine_batches` stops passing `analysis["ai_score"]` into aggregation. It still attaches `analysis` for coverage and diagnostics. |
| [`grading/reporting.py`](../../../src/grading/reporting.py) | `ai_score` = `parsed["ai_score"]`; `code_score` = `parsed["code_score"]`. `_reported_ai_score` and its `_has_explicit_artifact` gate are deleted. `detector_ai_score` stays; `detector_code_score` is added beside it so both measurements remain readable. `evidence` and `deductions` become the judge's entries parsed from `content` (D5); `human_evidence`, `metrics`, `coverage`, and the raw `analysis` stay as diagnostics. The whole pass is extracted as `annotate_all()` so the PDF command reuses it rather than copying it. |
| [`grading/flags.py`](../../../src/grading/flags.py) | `_judge_above_evidence` becomes flag-only; `_divergence` becomes flag-only; `_flag_unsupported` no longer claims the score was discarded; `_has_explicit_artifact` and `_reported_ai_score` are removed. `JUDGE_ABOVE_EVIDENCE`, `MODEL_DISAGREES`, `UNSUPPORTED_CITATION`, `EVALUATION_FAILED`, `HIGH_AI_USAGE`, `LOW_COVERAGE`, `INCOMPLETE_EVIDENCE`, `DUPLICATE_REPORT`, `SCORE_SATURATION` all survive. |
| [`grading/runtime.py`](../../../src/grading/runtime.py) | The console summary keeps printing both numbers; the detector line is labelled as a measurement. No logic change. |
| [`pdf/cli.py`](../../../src/pdf/cli.py) | Re-annotate the loaded projects through `Reporting.annotate_all()` by default, printing that it refreshed; `--no-refresh` renders the stored annotation unchanged. Nothing is written back to the input. |
| [`pdf/context.py`](../../../src/pdf/context.py) | No code change needed for the scores — `evidence_items` already prefers `details["evidence"]`, which now holds the judge's citations. Its docstring is corrected, and it no longer prints the detector's human traces under the judge's number. |
| `src/grading/README.md`, `src/pdf/README.md`, `src/contracts/README.md`, `src/README.md` | The withheld-score paragraph, the "runtime consumers may reject or replace scores" sentence, and the "the detector's evidence and deductions" sentence are rewritten to describe judge-owned scores and flag-only divergence. |
| this spec, [`docs/README.md`](../../README.md) | The new record is indexed. |

Not touched: [`detection/`](../../../src/detection/README.md), the model workspace, the prompt
contract, the window/batching machinery, `scores.json`'s schema beyond the two fields above.

## 5. Consequences, stated plainly

**5.1 The panels follow the score.** With D1 in force, a page that prints the judge's 74% above a
list of the *detector's* findings would be reading two different evaluations as one. So the AI
evidence panel prints the judge's cited entries (recovered from the report's `Evidence:` line)
and the deductions panel prints the judge's `Deductions:` entries. The detector's own findings,
weights, and citation strings remain in the stored `analysis` and are still written to
`scores.json`. Strike D5 and the panels keep printing the detector's findings under the judge's
number; D1–D4 are unaffected either way.

**5.2 The strongest-first ranking stops applying to new rows.** `strongest_first` orders the
evidence panel by the detector's finding weights, keyed on the detector's citation wording. The
judge's entries have no matching weights, so they print in the judge's own order. That is the
honest outcome — the judge's order is the order it considered the evidence — but it means the
ordering added on 2026-10-14 now serves only rows that still store detector citations. The cap
and the remainder note are unaffected, and the PDF tests that exercise ordering keep passing
because their fixtures store detector citations.

**5.3 The magnitude guard is gone.** Unsupported citations still produce a flag and a finding,
but the report no longer lowers the score for them. A judge entry quoting text that was never
shown now publishes its number with a warning attached rather than losing to a heuristic. This
is the deliberate cost of D2, and it is why `UNSUPPORTED_CITATION` needs to stay prominent in
the PDF's flag list.

**5.4 The judge's own calibration is now the whole story.** The restart design measured one
repository where the serving model returned 40.0% against a computed truth of 7.5% — landing
exactly on the 40% flag threshold, the band collapse it was written to diagnose. Under D1 that
number is published. `SCORE_SATURATION` (near-uniform scores across a run) and `HIGH_AI_USAGE`
are the remaining signals that a judge's number is not separating projects, and §7.2's baseline
work is what will replace this judgement with a measured one. D1 does not create that problem;
it stops a heuristic from hiding it.

**5.5 Nothing about the judge's input changes.** Because D6 keeps the measurement block, scores
from before and after this change remain comparable, and the judge is not asked to answer
questions it was not trained on.

## 6. Ordered plan

1. **Aggregation stops substituting.** Remove the `detector_ai_score` parameter and the
   overwrite in `combine_reports`; update `combine_batches` and the three test call sites.
2. **Reporting publishes the judge.** `ai_score`/`code_score` from the parsed report;
   delete `_reported_ai_score` and `_has_explicit_artifact`; add `detector_code_score`.
3. **Flags become advisory.** Reword `_judge_above_evidence`, `_divergence`,
   `_flag_unsupported`, and the `EVALUATION_FAILED` finding.
4. **Panels follow the score (D5).** Parse the judge's `Evidence:` and `Deductions:` entries
   from `content`; correct the `pdf/context.py` docstring.
5. **Documentation.** Rewrite the affected passages in the four READMEs and index this spec.
6. **Tests, then verify.** Update the assertions that encoded substitution as correct
   behaviour, add tests for the new behaviour, and run the offline suite plus a real
   `scores.json` through the PDF command.

## 7. Tests

| Test | Asserts |
| --- | --- |
| `TestReportedJudgeScore` (rewritten) | A judge score 72 points above a 2% measurement is **published** as 74, carries `JUDGE_ABOVE_EVIDENCE` and `HIGH_AI_USAGE`, and the finding names both numbers. A judge score with no detector analysis is published. |
| `TestQualityScoreIsTheJudges` (new) | A row whose judge scored 68% quality over a detector measurement of 71% publishes 68, flags `MODEL_DISAGREES`, and keeps 71 in `detector_code_score`. |
| `TestNoAggregationSubstitution` (new) | `combine_reports` with a batch full of unsupported citations returns the weighted judge mean, not a detector value, and still lists `unsupported_citations`. |
| `TestJudgeEntryPublication` (new) | A single-batch report publishes the judge's `Evidence:` entries in `evidence` and its `Deductions:` entries in `deductions`, while `analysis.ai_evidence` keeps the detector's. |
| `TestUnparseableReport` (updated) | An unparseable report publishes `ai_score`/`code_score` as `None` with `EVALUATION_FAILED`, and no detector number appears in either field. |
| `TestRefreshOnRender` (new) | `annotate_all` returns the judge's numbers for a stale row without touching a file; rendering a stale file draws 74% and 68% and not the detector's 90%; `--no-refresh` draws the stored 2% and 90% instead; the input file's bytes are unchanged by a refresh; an unparseable row still reports no score. |
| `tests/test_report.py` fixtures (updated) | The scored fixture's reported 71% becomes the judge's 68% and the detector's 71% moves to the diagnostic field. |
| `tests/test_scoring.py` (unchanged) | The live summary already prints both numbers in separate columns and needs no change. |

## 8. Definition of done

- No path in `src/` assigns `ai_score` or `code_score` from a `detection` value.
- Every successful row carries the judge's two numbers, the detector's two numbers
  (`detector_ai_score`, `detector_code_score`), and any divergence flag. No module under `src/`
  reads a `detector_*` score back to compute a reported one.
- An unparseable repository reports `null` and `EVALUATION_FAILED`, never a detector score.
- `uv run pytest` passes offline, and `uv run hackgrader-pdf` renders the existing
  `scores.json` with the judge's numbers on every page.
- The four READMEs describe judge-owned scores and flag-only divergence, with no sentence
  still claiming a detector substitution.

## 9. Risks

- **Score inflation on clean, tidy student projects.** This is the specific failure the
  substitution was built to catch, and D1 removes the automatic answer to it. Mitigated by
  keeping `HIGH_AI_USAGE`, `JUDGE_ABOVE_EVIDENCE`, `SCORE_SATURATION`, and the detector's
  numbers in every row, so a reviewer can still see a number running above its measurement —
  the reviewer now decides instead of the pipeline.
- **Threshold pile-up.** A judge that clusters on round values around the 40% flag threshold
  makes `HIGH_AI_USAGE` fire on band shape rather than on code. Unchanged by this work, and the
  reason §7.2 of the restart design (measure the baseline before retraining) stays the next
  milestone.
- **Losing the ranking in the PDF panels.** Accepted in §5.2; the alternative is ordering the
  judge's citations by a weight that was never computed for them.
- **Scope.** This is a reporting and aggregation change. The detector, the prompt contract, the
  batching, and the model workspace are explicitly out of scope, and no detector code is
  deleted — it stays the measurement that lets a reviewer audit the judge.

## 10. Measured effect on the stored run

The repository's 29-project `scores.json`, re-annotated with the implemented code and rendered
through `hackgrader-pdf` (31 pages, one cover plus at least one page per project):

| Observation | Before | After |
| --- | --- | --- |
| AI scores differing from the stored value | — | 17 of 29, every one of them upward |
| Quality scores differing from the stored value | — | 24 of 29 |
| Projects carrying `JUDGE_ABOVE_EVIDENCE` | 0 (the substitution hid it) | 16 |
| Projects reported above the 40% flag threshold | 0 | 15 |
| Reported AI scores | 0–20 on every project | 11 of the 26 scored projects land in a 74–78 band |

The stored file was re-annotated in place by the decision recorded in §3, with the previous
annotation kept beside it as `scores.stale.json`. All 29 rows now carry `detector_code_score`,
and re-running `hackgrader-pdf` afterwards produced a file identical to a fresh re-annotation,
so the JSON and the printed report agree rather than drifting apart.

Every one of the 16 gap flags is a project the previous code published at the detector's number
— `Nora070/LSU-Hackathon` 6 → 74, `stanlsu/hackathonhuzz` 7 → 77, `dtaing11/Hackathon-Wics-2026`
8 → 78, and so on — and on each of those the detector had measured between 6 and 17. The
quality column moved the other way in places, where the judge's read was stricter than the
detector's (`davidzhong39/A-Project-of-Happiness` 100 → 85).

### 10.1 The 74–78 band is a rubric ceiling, not a collapse

An earlier reading of this run called the eleven-project plateau a band collapse. Splitting the
corpus on the only label independent of both scorers — a repository naming an AI tool as its own
author — does not support that reading, and the spread of the two scorers contradicts it:

| Scorer | Range | Mean | Distinct values |
| --- | --- | --- | --- |
| Judge, AI involvement | 0–78 | 46.4 | 18 of 26 |
| Detector, AI involvement | 0–20 | 10.5 | 17 of 26 |
| Judge, code quality | 83–98 | 85.7 | 6 of 26 |
| Detector, code quality | 54–100 | 84.7 | 13 of 26 |

**Zero of 29 repositories name an AI tool anywhere in their stored output**, and no detector
artifact finding exists in the corpus either. The prompt reserves the top of the scale — 85 and
above — for "a quoted tool naming itself as the author", so a ceiling at 78 is the rubric
working as written: nothing in this corpus can legitimately exceed the 50–85 band. Rank
agreement between judge and detector on the AI column is +0.42, so the plateau is not a
constant either; it is the top of the band the prompt allows.

What the plateau genuinely costs is resolution at the top. The eleven land at 74, 75, 76, 76,
77, 77, 78, 78, 78, 78, 78 while the detector still separates them across 6–17, and they are
the largest repositories in the corpus (274–936 source lines, 15,374–33,345 characters) against
a lower half of 0–270-line projects. The judge is tracking volume into a ceiling.
`SCORE_SATURATION` did not fire: it requires scores at or above 90, so a plateau at 76 is
invisible to a flag that reads only as a fixed threshold.

A second independent check fails for both scorers. The prompt states that human traces — TODO
notes, debug output, mixed formatting — belong in the 0–15 band, and the detector counts them.
Five projects carry one or two TODO notes and the judge scored them 5, 57, 67, 78 and 78, the
same range as projects carrying none, with two of its five highest scores belonging to projects
that do have them. On this corpus the human-trace signal does not move either scorer's number.

### 10.2 The column that does not discriminate is code quality

The compression is in the score this change handed to the judge. Across 26 scored projects the
judge returned six distinct quality values — 83 once, 84 three times, **85 twelve times**, 86
eight times, 87 once, and one 98 — so **25 of the 26 fall inside a four-point range and 12 are
exactly 85**. The detector's quality column spans 54–100 across 13 distinct values and lands on
exactly 85 only once, so its 85 is not a mode. Mean absolute difference between the two is 6.2
points, and their rank agreement is **+0.02** — no ordering agreement at all, so at most one of
them is tracking anything.

This is a real cost of D1, recorded rather than argued away: the quality column moved from a
measurement with a 46-point spread to one that is effectively the constant 85, and it is the
column §3 of the restart design expects to carry "how good is this code". Under the old code
that column was the detector's, always. Answering the band question therefore points at the
quality score, not at the saturation threshold: the next work is to find out whether 85 as a
mode is a stable model behaviour on tidy student projects or a scale anchor, which needs a
corpus with known provenance — the §4 cohort of that design — because this one contains no
labelled example of any kind.

On the PDF itself, one project's own page was checked by decoding its drawn text: it prints the
judge's 85%, does not print the detector's 100%, contains no occurrence of the word "detector",
and draws both the judge's evidence entries and its deduction lines under the new panel labels.
