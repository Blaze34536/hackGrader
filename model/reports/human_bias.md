# What the served judge says about code no agent wrote

Validation measures agreement with generated teaching labels. This measures the thing those
labels were standing in for: the AI percentage the application reports for repositories that
ended before coding agents existed.

## Method

`scripts/measure_human_bias.py` grades each repository through the application's own path
(`grading.runtime.Model`, the deployed tag from `deployed_model.json`, every evidence batch,
line-weighted aggregation), then records the score the report states.

Dormancy is checked against GitHub's `pushed_at`, not assumed: a candidate whose last push is
on or after the cutoff is skipped and listed. Twenty repositories survived with a last push
before 2023-01-01, nineteen of them before 2022-01-01. Languages: C, JavaScript, Go, Python.

The detector's own reading (`detection.analyze`) is recorded beside each score. It is not
the judge's target and not on the same scale, but it answers a separate question: does the
model merely reproduce the tool's reading, or does it diverge from it?

## Result

| | judge | detector |
| --- | --- | --- |
| repositories | 20 | 20 |
| mean | **82.3** | 16.4 |
| median | 84 | 14 |
| range | 67–89 | 0–30 |
| at or above 40% | **20 / 20** | 0 / 20 |
| at or above 70% | 19 / 20 | 0 / 20 |

Mean gap between the two readings: **65.9 points**.

Every repository in the sample is dormant before any agent existed, so its true AI share is
about zero. The judge calls all twenty AI-written and nineteen of them strongly so.

## The evidence it cites

From `tj/co` (last push 2020-12-15, 83%):

```
Rank 1: 7 JSDoc blocks appear in the source, for example `/** slice() reference. */` in
        index.js, this supports the score because of 7 AI-generated JSDoc blocks …
Rank 2: 0 line(s) are comments, for example 'var assert = require('assert');' in
        test/arguments.js, and 16 of 19 files have this pattern
Rank 3: 0 of 2 source lines are TODO/FIXME notes ... this supports the AI score
```

The current system prompt in `src/contracts/prompts.py` forbids exactly these readings:
"Ordinary structure, formatting, comments, docstrings, or a tidy README are weak or
non-specific signs ... must not by themselves justify a high score", and "Do not infer
authorship from missing TODOs or other absent human traces."

The deployed adapter predates that prompt and those target constraints. Its data taught it
the opposite rule — the build that trained it weighted README structure and docstrings
heavily and inverted "human traces", so absent TODOs read as machine-like. The model follows
the data over the prompt.

## What this changes

1. The model is a second defect, not only a faithful copy of the labeler. Its teacher reads
   16.4 on this sample; the model reads 82.3. Rebuilding the targets is necessary but does
   not by itself produce calibration.
2. Validation could not have caught this: its rows are scored by the same function that
   produced the labels, and none of them land in the 25–60 band where an over-reading
   answer would be wrong.
3. The fix is therefore both: rebuild repository targets from commit attribution (the
   harvester's manifest) and retrain so the prompt's rule is in the weights, not just in the
   prompt text.

## Limits

- "Human-written" here means dormant before 2023, checked at repository level. A repository
  could still vend generated code; commit-level attribution is what the harvester adds and
  was not applied to this sample.
- Generation runs at temperature 0.2 with fixed seeds, so a rerun of one repository can move
  a few points. The conclusion rests on twenty independent repositories clustering at 83–89,
  not on any single number.
- Some evidence batches produced no valid report (up to 3 of 15 on the largest views) and
  were dropped from the aggregate, as they are in production.
- The detector's scale is not a ground truth either. Its 16.4 is reported as a comparison,
  not as a calibration claim.

Artifacts: `human_bias.jsonl` (judge scores and reports), `human_bias_detector.jsonl`
(detector readings), `human_bias_run.log` (the run).
