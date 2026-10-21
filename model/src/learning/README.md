# Learning, evaluation, and release

The `learning/` package validates model inputs, fine-tunes adapters, selects checkpoints, measures holdouts, and coordinates release. Its central safety rule is that **validation guides training; the independent test does not**.

- **Training rows** update model weights.
- **Validation rows** select checkpoints and may guide further training.
- **Independent-test rows** are reserved for a final, gated evaluation. A failure blocks release; the scored test must not be used to continue training or tune thresholds.

Generated target scores are heuristic teaching labels, not human-verified authorship judgments. Agreement with them is not proof of accuracy.

## Command reference

Run commands from `model/`; use `uv run src/cli.py <command> -h` for full options.

```bash
# Validate configured inputs and fresh output paths; do not load or train a model.
uv run src/cli.py train --preflight-only

# Train using the configured combined candidate.
uv run src/cli.py train

# Continue an existing run without resetting its original run budget.
uv run src/cli.py train --resume-checkpoint outputs/<run>/round-01/checkpoint-250

# Start an isolated run and fresh time cap from a saved adapter.
uv run src/cli.py train --fork-checkpoint outputs/<run>/round-01/checkpoint-250

# Inspect checkpoints on validation without saving weights or registering a model.
uv run src/cli.py score -r outputs/<run> --all-checkpoints

# Preview deployment commands only.
uv run src/cli.py deploy --dry-run
```

The model-side learning and training-data suites are no longer part of the repository, so changes here are verified through the application contracts under `tests/` instead. The package can still be linted from the repository root:

```bash
uv run --project model --group dev ruff check model/src/learning
```

### Run modes

| Mode | Behavior |
| --- | --- |
| `--preflight-only` | Validates inputs and output destinations, reports a run estimate, and exits before model loading. It does not train, score the independent test, merge, or deploy. |
| `--continue-run` | Inspects the selected or newest run and takes the one safe action: resume training, rebuild an interrupted release, report an already-released run, or explain why it cannot continue. Without `--run-id` it skips runs that saved no checkpoint or claimed no test. A resume reuses the data files recorded in the run's metadata unless the command names data. `--preflight-only` reports the decision without loading a model. |
| `--resume-checkpoint` | Continues inside the original run and its remaining time cap. It refuses a run whose independent test was already claimed; use `--continue-run` or `recover` for that. |
| `--fork-checkpoint` | Creates a separate run and time cap from a saved adapter while preserving the source run. It may train unless paired with `--release-only`. |
| `--release-only` | On a fork, evaluates saved candidates and performs the release sequence without an optimizer round. It is not a no-op: it may load a model and score holdouts. |
| `--strict-gates` | Disables provisional release and stops when adaptive validation tuning bottoms out. It does not weaken release requirements. |
| `--fresh-repository-test` | Restricts the final test to real-repository rows with URLs not present in training or validation. |

`--fresh-repositories <file>` is a different option: it asks `train` to build a new candidate from the supplied repository list and requires `--fork-checkpoint`. The command implies strict gates and the fresh-repository-test filter. To inspect a built candidate without training, prefer running the data builder separately, then use `--dataset <candidate> --fork-checkpoint <checkpoint> --fresh-repository-test --strict-gates --preflight-only`.

### Multi-GPU

`train` defaults to `HACKGRADER_GPUS=auto`, which trains on every visible CUDA GPU when the platform provides NCCL; set it to a number to pin the worker count, or to `1` to stay on one GPU. The command relaunches itself under `torchrun` with one worker per GPU, and each worker holds a full 4-bit model replica, so the model has to fit on every GPU on its own. Worker 0 alone scores validation, spends the independent test, merges, and registers; the other workers train the same rounds and follow worker 0's plan and decision files in `outputs/<run-id>/workers/`, then exit once training ends. Because validation can run for hours, decisions travel through those files rather than through collectives. Gradient accumulation is divided by the worker count (down to one), so the optimizer batch matches a single-GPU run and the extra GPUs buy speed instead of a different learning problem. A single GPU, a platform without NCCL (including Windows), or `HACKGRADER_GPUS=1` trains on one GPU and prints why. `--preflight-only` and `--release-only` never relaunch, and `--fresh-repositories` builds its data before training, so that invocation is single-GPU.

## Evaluation and release sequence

1. Preflight loads and verifies the selected dataset, split sizes, provenance, coverage, and output paths before GPU dependencies are loaded.
2. A training run estimates its budget, creates fresh run outputs, and performs bounded optimizer rounds. The default total wall-clock cap is 20 hours, including validation, final testing, and release reserves. A round defaults to at most 500 steps; up to four candidates are scored per round.
3. Validation selects the checkpoint and can guide further rounds. New rounds continue from selected adapter weights with a fresh optimizer.
4. The selected checkpoint receives at most one final-test attempt in the gated release flow. A passing validation and test can allow normal release. If validation still fails at the cap, a passing test may allow only the separate `hackgrader-provisional` tag unless strict gates are enabled. A failed test blocks merge and registration.

The diagnostic `score` command is read-only and validation-only; attempts to select the independent test are rejected. Use it for validation diagnostics, not to replace the release flow. Training and manual validation scoring save row-level diagnostic reports beside the run's checkpoints; see the [scoring reference](scoring/README.md) for contents and privacy notes.

## Configuration reference

Values are read at process startup. The full defaults and cross-setting checks are in [`config.py`](config.py), [`config_validation.py`](config_validation.py), [`training/trainer_config.py`](training/trainer_config.py), and [`../data/config.py`](../data/config.py).

| Area | Common variables |
| --- | --- |
| Dataset inputs | `HACKGRADER_DATASET_FILE`, `HACKGRADER_TRAIN_FILE`, `HACKGRADER_VAL_FILE`, `HACKGRADER_TEST_FILE` |
| Output paths | `HACKGRADER_OUTPUT_DIR`, `HACKGRADER_ADAPTER_DIR`, `HACKGRADER_MERGE_DIR`, `HACKGRADER_RUN_ID` |
| Base model and budget | `HACKGRADER_BASE_MODEL`, `HACKGRADER_HOURS`, `HACKGRADER_MAX_STEPS_PER_ROUND`, `HACKGRADER_MAX_GATE_CANDIDATES`, `HACKGRADER_GPUS` |
| Runtime and tokens | `HACKGRADER_BATCH`, `HACKGRADER_ACCUM`, `HACKGRADER_MAX_LEN`, `HACKGRADER_ASSISTANT_ONLY_LOSS` |
| Training dynamics | `HACKGRADER_LORA_LR`, `HACKGRADER_LORA_DROPOUT`, `HACKGRADER_WEIGHT_DECAY` |
| Evaluation size | `HACKGRADER_CHECK_ROWS`, `HACKGRADER_MIN_TEST_ROWS`, `HACKGRADER_MIN_TEST_CLASS_ROWS` |
| Release gates | `HACKGRADER_TEST_MIN_PRECISION`, `HACKGRADER_TEST_MIN_RECALL`, `HACKGRADER_TEST_MIN_SPECIFICITY`, `HACKGRADER_HOLDOUT_MAE_LIMIT`, `HACKGRADER_MIN_HOLDOUT_SPREAD`, `HACKGRADER_MIN_HOLDOUT_SEPARATION` |

## Package references

| Package | Responsibility | Reference |
| --- | --- | --- |
| `pipeline/` | Coordinates preflight, training, evaluation, and release. | [Pipeline contract](pipeline/README.md) |
| `preflight/` | Loads and checks datasets and output paths. | [Preflight checks](preflight/README.md) |
| `training/` | Runs bounded optimization rounds, checkpoint selection, and recovery. | [Training rounds and selection](training/README.md) |
| `holdout/` | Samples and evaluates holdouts and enforces release gates. | [Holdout and uncertainty behavior](holdout/README.md) |
| `scoring/` | Provides read-only saved-checkpoint diagnostics. | [Scoring interface](scoring/README.md) |
| `deployment/` | Verifies artifacts, converts a merge, and registers an Ollama tag. | [Deployment interface](deployment/README.md) |

### Top-level modules

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Adds the shared `contracts` source path so the package can import it. |
| `config.py` | Configures fine-tuning, evaluation sizes, outputs, and release checks. |
| `config_validation.py` | Validates training, data, and release-gate configuration values. |
| `continuation.py` | Chooses the one safe next action for a run: resume, rebuild the release, report it done, or explain why not. |
| `run_artifacts.py` | Reads a run's claim marker, checkpoints, selection, adapter, stamp, and recorded inputs from disk. |
| `distributed.py` | Launches the torchrun job and shares worker 0's round plans and decisions. |
| `inputs.py` | Prepares model-visible conversations and proves assistant-only training loss. |
| `runtime.py` | Loads GPU dependencies only after training data has passed preflight checks. |
| `resume.py` | Validates and scores saved adapters before restarting a training process. |
| `resume_fork.py` | Validates outputs for a fresh-budget run forked from a saved adapter. |
| `resume_outputs.py` | Protects existing checkpoints and active inputs from resume outputs. |
| `resume_runtime.py` | Resolves an existing run and enforces its original wall-clock budget. |

Output paths must be fresh, stay in the model workspace, and remain isolated from active inputs and protected files. Never remove existing outputs to bypass a path check. See the [model workspace overview](../../README.md) and [candidate data reference](../data/README.md) for setup and dataset details.
