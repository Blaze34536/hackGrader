# Optimizer training and checkpoint selection

The `learning.training` package runs bounded fine-tuning rounds, selects saved adapters using validation, and preserves usable checkpoints when a round is interrupted. The higher-level [learning pipeline](../README.md) owns preflight, the final independent test, and release.

## Round lifecycle

1. The pipeline establishes the run deadline and reserves time for validation scoring, final testing, and release.
2. The loop calculates a step budget that fits the remaining time. Each round has its own output directory and a fresh optimizer; a continuation round starts from the best validation-scored adapter from the prior round.
3. The trainer saves checkpoints at evaluation intervals. The selector scores a bounded number of candidates on validation and retains the strongest selection.
4. If a round is interrupted, recovery checks available saved checkpoints using validation only. The run keeps the best usable selection and does not merge or deploy an unvalidated result.
5. Strict-gate runs stop if adaptive learning-rate tuning reaches its floor without passing validation. They do not use provisional release.

The default round limit is 500 optimizer steps, validation/save intervals default to 50 steps, and checkpoint selection considers at most four candidates per round. The overall run cap and reserved phase budgets are configured in [`learning/config.py`](../config.py); all limits can be overridden through the documented `HACKGRADER_*` settings.

Validation may guide checkpoint selection and future rounds. The independent test is not consulted by this package; it is reserved for the one-time final evaluation in the [pipeline](../pipeline/README.md). Do not use test results to tune training, thresholds, or checkpoint selection.

## Module map

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Package marker; the modules above are imported directly. |
| `fit.py` | Configures and runs one bounded supervised fine-tuning round. |
| `round.py` | Isolates a round and reports runtime failures to the coordinator. |
| `loop.py` | Allocates time across rounds and continues from selected adapter weights. |
| `budget.py` | Calculates how many whole evaluation intervals fit the remaining budget. |
| `checkpoints.py` | Bounds candidate selection and scores checkpoints on validation. |
| `selection.py` | Compares a new round's validation result with the previous selection. |
| `adaptation.py` | Adjusts the next learning rate from validation progress and enforces the strict-mode floor. |
| `recovery.py` | Recovers saved candidates after an interrupted round using validation only. |
| `callbacks.py` | Reports memory use and stops at deadlines or configured early-stopping patience. |
| `trainer_config.py` | Configures evaluation/checkpoint behavior and quantized model placement. |
| `artifacts.py` | Creates run metadata, verifies merged weights, records provenance, and estimates runtime. |

For the checkpoint-callback reinitialization needed after Unsloth switches back into training mode, see [`trainer_config.py`](trainer_config.py).

The model-side learning and training-data suites that verified these workflow and safety contracts without performing a real fine-tuning run are no longer part of the repository, so changes here are verified through the application contracts under `tests/` instead.
