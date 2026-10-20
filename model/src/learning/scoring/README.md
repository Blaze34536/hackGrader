# Saved-checkpoint scoring

The `score` command reports metrics for adapters already on disk. It is a diagnostic path: it does not train, save adapter weights, merge a model, or register an Ollama tag. It defaults to validation and rejects independent-test scoring so the test remains reserved for the gated release flow.

During training, a separate bounded selection path scores candidates on validation to choose continuation weights. The command-line diagnostic does not replace that selection or release process.

## Commands

Run from `model/`:

```bash
uv run src/cli.py score -h
uv run src/cli.py score -r outputs/<run-id> --all-checkpoints
uv run src/cli.py score -r outputs/<run-id> --checkpoint-step 350
uv run src/cli.py score -r outputs/<run-id> --checkpoint-step 350 \
  --export-predictions outputs/checkpoint-350-validation.predictions.jsonl
uv run src/cli.py score \
  --prediction-file outputs/checkpoint-350-validation.predictions.jsonl \
  --sweep-thresholds 35,36,37,38,39,40,41,42,43,44,45
```

The default is the newest checkpoint in the newest run, on a bounded validation sample. Choose a run with `--run-dir`, several saved checkpoints with `--all-checkpoints`, or one or more steps with `--checkpoint-step`. `--limit` and `--batch-size` configure inference. Prediction export is for one validation checkpoint and contains prediction metadata and source identifiers, not repository prompts. Threshold calibration consumes a saved validation prediction file; it does not change release gates.

The diagnostic CLI rejects `--split test` and `--split both`. Do not try to evaluate or tune on the independent test through this command. That split is available only once through the gated release workflow.

## Validation failure artifacts

Every validation candidate scored during training is written as `validation-checkpoint-<step>.jsonl` inside its run directory, including per-row target/prediction classes, parsing failure categories, generated report text, and gate metrics in the first-line metadata. If the same logical step is scored more than once, a numbered suffix preserves the earlier artifact. These files never include repository prompts or test predictions. Generated reports may still quote code from the validation view, so keep the output directory private and review it as potentially sensitive data. Diagnostic-write failures are logged as warnings and do not change gate decisions.

A manual `score` command also writes a validation diagnostic artifact by default. Use `--export-predictions <path>` if you additionally want the compact calibration export; that export deliberately omits raw generated text.

## Inference cost and batch size

Validation scoring is autoregressive generation, so it is the slowest phase of a run and the phase that competes with the window reserved for it. Measured on the 16 GB RTX 5080 used for the 2026-10 runs, a 50-row validation sample with roughly 1100-token prompts and reports of 150-241 tokens costs about 28 seconds per row at `--batch-size 1`, holding peak allocation near 11 GB. The same sample at `--batch-size 4` climbed to about 15.7 GB of the 16.3 GB card and cost roughly three times more per row, because the allocator then runs against the card ceiling. The default is therefore 1, and generation releases the allocator cache after every batch. At that rate a four-candidate sweep costs about 1.6 hours instead of most of a day, which is what lets it fit the validation-scoring reserve that `HACKGRADER_HOURS` sets aside. Raise `--batch-size` only on a card with measured headroom.

Generated reports are capped at `MAX_NEW_TOKENS` (320). A report truncated before its summary is unparseable, and unparseable rows count against the `unparsed` gate, so the cap is set well above the longest target (231 tokens) and the longest generated report observed so far (241 tokens). Rows that finish normally emit an end-of-sequence token and stop early, so the extra headroom costs nothing on them.

## Module map

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Re-exports the scoring entry points and split labels. |
| `discovery.py` | Finds runs and checkpoints and loads scoring splits. |
| `measure.py` | Scores one checkpoint and ranks candidates. |
| `sweep.py` | Scores several adapters while reusing a resident base model. |
| `report.py` | Formats checkpoint metrics and comparisons. |
| `commands.py` | Validates options and coordinates the read-only score command. |
| `exports.py` | Writes validated prediction metadata and validation cutoff reports. |
| `cli.py` | Exposes the score command and compatibility imports. |
| `diagnostics.py` | Persists validation-only gate diagnostics beside a training run. |

Numbered rounds use distinct logical step ranges; for example, `round-02/checkpoint-50` is not the same checkpoint as `round-01/checkpoint-50`. The training pipeline's validation selection is the source of truth for continuation, final testing, and deployment. See [holdout gates](../holdout/README.md) for metric interpretation and the [pipeline reference](../pipeline/README.md) for phase order.
