# Holdout evaluation and release gates

The `holdout/` package measures model predictions against the dataset's target scores and checks whether a checkpoint meets release requirements. It does not certify that generated teaching labels are true authorship judgments.

## Keep the split roles separate

- **Validation** may be scored repeatedly to choose checkpoints and guide further training.
- **Independent test** is reserved for the gated release flow. It can be attempted once for the selected candidate; its result may block release but must not guide further training or tuning.
- **Fresh-repository test** further filters the independent test to real-repository rows whose canonical URLs do not occur in training or validation. AIGCodeSet, synthetic, and artifact rows do not count toward its coverage.

Preflight checks coverage before model loading. By default, test coverage requires at least 50 distinct selected prompts with at least 10 below and 10 at/above the 40% class threshold. Additional per-split size and score-band gates apply during dataset validation. The fresh-repository builder can reserve at least 50 real-repository prompts and at least 10 in each class, but final training preflight applies the real test-only filter and checks the resulting sample again.

## Metrics and interpretation

The evaluator reports score error and classification behavior at the configured 40% threshold, including precision, recall, specificity, score spread, and separation where applicable. Release floors are configured in `learning/config.py`; a missing requirement is a failure, not a reason to silently change the threshold.

Uncertainty intervals help describe limited samples; they do not waive deterministic gates. Proportion intervals use Wilson bounds with source-cluster adjustment where enough independent clusters exist. MAE, spread, and class-separation uncertainty use deterministic source-cluster bootstrap estimates, resampling rows from the same source together. Fewer than ten independent source clusters produce conservative full-domain intervals. An interval that overlaps a gate is inconclusive; only the configured deterministic checks determine pass/fail.

Generated teaching-target agreement is not human-verified authorship accuracy. Interpret metrics against the labels and sources they actually measure.

## Module map

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Re-exports the coverage, evaluation, gate, and enforcement entry points. |
| `coverage.py`, `sampling.py` | Check sample sufficiency and choose unique prompts. |
| `generation.py` | Runs bounded inference and records predictions with source lineage. |
| `metrics.py`, `class_metrics.py`, `metric_calculations.py` | Compute score error, classification counts, and cluster-aware estimates. |
| `intervals.py`, `lineage.py` | Calculate uncertainty and group shared source identities. |
| `gates.py`, `coverage_violations.py`, `threshold_violations.py` | Describe release requirements and collect failures. |
| `evaluation.py` | Measures a holdout and gathers gate failures. |
| `enforcement.py` | Stops callers that require passing results. |

A failed final test blocks normal merge and registration. A validation-failing run at its cap may qualify only for the separate provisional tag if that policy is enabled and its single test passes; strict-gate mode disables provisional release.
