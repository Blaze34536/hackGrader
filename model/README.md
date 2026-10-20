# Model training workspace

The `model/` project builds reviewable candidate data, fine-tunes HackGrader adapters, evaluates saved checkpoints, and releases a verified Ollama model through explicit gates. It uses the shared grading prompts and report contract in the repository-root `src/` tree.

## Before you start

- The model workspace requires **Python 3.12**; the application project at the repository root requires Python 3.14 or newer.
- Install dependencies from the repository root: `uv sync --project model --group dev`.
- Run model commands with `model/` as the working directory.
- Training and model inference require a compatible CUDA GPU. Data building may need network access, source URLs, model/tokenizer downloads, and a configured repository cache.
- Candidate-generation inputs are user data. Review source lists and generated labels before using a candidate for training. Detector-derived targets are teaching labels, not verified authorship judgments.

## Fresh clone: build the data, then train

A clone carries the code, the configuration, and the hand-written repository lists, but not the generated inputs and artifacts. `data/repository_labels.jsonl`, `data/cohort.jsonl`, `data/candidate_repos.txt`, `data/dataset.candidate.jsonl`, `outputs/`, `models/`, and `llama.cpp/` are all ignored by Git. Training supervises the repository head from the accepted label manifest, so on a new machine that manifest has to be rebuilt before anything can train. This is the whole sequence, in order.

Prerequisites: Python 3.12 (installed by `uv`), an NVIDIA GPU with at least 16 GB of VRAM and a driver for CUDA 12.8, roughly 60 GB of free disk (the 4-bit base model download, the 16-bit merged weights, and the quantized GGUF), network access, a `GITHUB_TOKEN`, a running Ollama, and a POSIX shell — Git Bash is enough on Windows.

```bash
uv sync --project model --group dev   # Python 3.12 environment with CUDA 12.8 torch wheels
cd model
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git   # GGUF converter, ignored by Git
export GITHUB_TOKEN=<token>           # PowerShell: $env:GITHUB_TOKEN = "<token>"
bash scripts/build_candidate_from_harvest.sh   # harvest labels, verify them, build the candidate
uv run src/cli.py train --preflight-only       # check the data and destinations; loads no model
uv run src/cli.py train                        # train, select, gate, merge, and register
```

**1. Environment.** The model workspace pins Python 3.12 and pulls `torch` and `torchvision` from the CUDA 12.8 wheel index. Run every command after the first from `model/`.

**2. Converter.** The release step at the end of training shells out to `llama.cpp/convert_hf_to_gguf.py`; the workspace's own dependencies already satisfy its imports, so the clone is the only setup it needs. Training registers an Ollama tag as its last action, so start Ollama before step 5 rather than discovering it is missing after twenty hours.

**3. Data.** `scripts/build_candidate_from_harvest.sh` harvests agent-attributed and human repositories, verifies the resulting manifest with the same fail-closed validator the builder uses, and writes `data/dataset.candidate.jsonl`. It is resumable: rerunning it after an interruption continues the cohort instead of starting over. One budget cannot always reach the repository goal, so `harvest --resume-to-goal` keeps starting fresh budgets and waits out GitHub's hourly reset until `--max-repositories` is reached. Without `GITHUB_TOKEN` the harvest runs at 60 core requests an hour and will not finish. The build also downloads the AIGCodeSet rows and fetches repository snapshots, so it needs network access.

If the existing checkout is available, copying `model/data/` across (about 33 MB) skips the harvest entirely; it holds the label manifest, cohort, and candidate dataset that a rebuild would reproduce.

**4. Preflight.** `--preflight-only` validates the dataset, the score-band and class floors, the input paths, and the output destinations, then stops before importing CUDA or Unsloth. It also prints the runtime estimate, which is what the default 20-hour cap is checked against.

**5. Train.** A normal run trains, selects checkpoints on validation, scores the reserved independent test once, and — only if both gates pass — saves the adapter, merges the weights, converts the GGUF, and registers the next `hackgrader:vN` tag. Fail a gate and nothing is merged or registered. Record the run id it prints (`outputs/<run-id>`) if you want to inspect the run afterwards. The default cap is 20 hours, so start it in a persistent session (`tmux` or an equivalent) instead of a terminal that a dropped connection would end.

**6. Confirm the release.** The deployment is part of step 5; these two commands only read it back:

```bash
HACKGRADER_RUN_ID=<run-id> uv run src/cli.py deploy --check   # verifies the served GGUF and prints its version
uv run src/cli.py score -r outputs/<run-id> --all-checkpoints # optional: validation scores per checkpoint
```

If the run stops before the independent test is claimed, continue it with `train --resume-checkpoint <checkpoint>`. If it stops after the test is claimed — for example during the merge — do not start a new run: the test is single-use, so `recover --run-id <run-id>` rebuilds only the merge, conversion, and registration from the adapter the run already saved.

## Command reference

Run from `model/`. Use `uv run src/cli.py <command> -h` for complete options.

| Goal | Command |
| --- | --- |
| List model commands | `uv run src/cli.py -h` |
| Harvest labels and build the default candidate | `./scripts/build_candidate_from_harvest.sh` |
| Harvest an audited repository cohort | `uv run src/cli.py harvest --since 2026-09-01 --until 2026-09-30 --labels-out data/labels.audited.jsonl` |
| Harvest to a repository goal across budgets | `uv run src/cli.py harvest --max-repositories 220 --resume-to-goal --labels-out data/repository_labels.jsonl` |
| Build with an existing audited manifest | `uv run src/cli.py data -r data/candidate_repos.txt --repository-labels data/repository_labels.jsonl` |
| Show data-builder options | `uv run src/cli.py data -h` |
| Validate inputs and paths without training | `uv run src/cli.py train --preflight-only` |
| Train with the configured candidate | `uv run src/cli.py train` |
| Resume the original run and time cap | `uv run src/cli.py train --resume-checkpoint outputs/<run>/round-01/checkpoint-250` |
| Fork a checkpoint into a new run | `uv run src/cli.py train --fork-checkpoint outputs/<run>/round-01/checkpoint-250` |
| Release a fork without an optimizer round | `uv run src/cli.py train --fork-checkpoint <checkpoint> --release-only` |
| Score the newest checkpoint on validation | `uv run src/cli.py score` |
| Compare saved checkpoints on validation | `uv run src/cli.py score -r outputs/<run-id> --all-checkpoints` |
| Preview deployment commands | `uv run src/cli.py deploy --dry-run` |
| Verify the configured Ollama model artifact | `uv run src/cli.py deploy --check` |
| Preview a rebuild of an interrupted release | `uv run src/cli.py recover --run-id <run-id> --dry-run` |
| Rebuild an interrupted release | `uv run src/cli.py recover --run-id <run-id>` |

The model-side learning and training-data suites are no longer part of the repository, so changes here are verified through the application contracts under `tests/` instead. The workspace can still be linted from the repository root:

```bash
uv run --project model --group dev ruff check model/src
```

## How the workflow protects evaluation

1. **Build:** the data builder validates report format, citations, provenance, source-disjoint splits, quotas, and token length before publishing a candidate.
2. **Preflight:** the training pipeline validates data and output paths before loading model/GPU dependencies. `--preflight-only` stops after checks and a runtime estimate; it does not train or deploy.
3. **Train and select:** train rows update weights. Validation selects checkpoints and may guide further training. A resume preserves its original run; a fork starts a separate run and cap.
4. **Final test:** the independent test is reserved for a final gated evaluation. Its result may block release but must not guide training or threshold tuning.
5. **Release:** a passing validation and test can allow normal release. A successful release takes the next sequential version from `data/model_versions.json` and registers it as `hackgrader:vN`; the ledger entry is written only after Ollama verifies the blob. A capped run that has not passed validation may qualify only for the separate provisional tag if its test passes; strict-gate runs disable provisional release. A failed final test blocks merge and registration.

The default total run cap is 20 hours, including reserved validation, final-test, and release time. Each optimizer round is capped at 500 steps by default. Generated teaching-label agreement does not establish real-world authorship accuracy.

A machine that loses power between the selected-adapter save and the merged-weight write leaves a run that cannot simply be restarted: the one-time independent test is already claimed, so `train` refuses to score it again. `recover --run-id <run-id>` finishes that run instead. It never re-scores the reserve test; it rebuilds only the merge, conversion, and registration from the adapter and validation diagnostics the run already wrote, and it refuses unless the saved adapter postdates the claim and still matches the recovered checkpoint tensor by tensor. The build records its own provenance in `merge_recovery.json` beside the merged weights.

## Where artifacts land

Training writes two trees, and the split is deliberate: `outputs/` is the run's working state, `models/` is the deliverable. Both are ignored by Git, so they never arrive with a clone. The run id in every path is the one printed when preflight or training starts (`[i] This run writes checkpoints to outputs/<run-id>`).

### `outputs/` — the run

| Path | Written when | Contents | Size |
| --- | --- | --- | --- |
| `outputs/<run-id>/run_metadata.json` | run start | base model, max length, step cap, start time, and `forked_from` for a fork | small |
| `outputs/<run-id>/round-<NN>/checkpoint-<N>/` | each evaluation point | the PEFT adapter for that step, its tokenizer, and `trainer_state.json`; a checkpoint is resumable, not a standalone model | about 280 MB |
| `outputs/<run-id>/validation-checkpoint-<N>.jsonl` | each validation scoring with row predictions | per-row gate evidence and generated responses; a repeat gets `-02`, then `-03` | a few MB |
| `outputs/<run-id>/.independent-test-attempted/` | immediately before the final test | an empty directory used as an atomic claim; its presence is what refuses a second attempt | — |
| `outputs/lora_adapter-<run-id>/` | release | the selected checkpoint's adapter in PEFT form (`adapter_config.json`, `adapter_model.safetensors`) | about 260 MB |

### `models/` — the model

| Path | Written when | Contents | Size |
| --- | --- | --- | --- |
| `models/qwen3_ai_detector_merged-<run-id>/` | release | 16-bit merged weights and their shard index, plus config, tokenizer, chat template, and `merge_stamp.json`; loadable by `transformers` without Unsloth or the base model | about 30 GB |
| `models/ai_detector_q8-<run-id>-<digest>.gguf` | release | quantization of the merged weights (`q8_0` by default), named with the first 16 hex characters of the adapter digest; this is the file Ollama serves | about 16 GB |

The merged directory is the artifact to copy to another machine: it is a complete `Qwen2ForCausalLM` checkpoint, while the adapter is only the delta and the GGUF is only for Ollama or llama.cpp. `merge_stamp.json` keeps the adapter hash, base model, training rows, `max_len`, merge time, and both gate results, which is what ties a directory to its run. A recovery adds `merge_recovery.json` beside it. Names carry the run id and adapter digest, so a later run does not overwrite an earlier one.

The two `data/` records are the exception to "generated": `data/deployed_model.json` names the tag the grader resolves, and `data/model_versions.json` records every released version. These are small, committed, and the only model-related files that belong in version control.

## Data and artifact safety

The default builder output is `data/dataset.candidate.jsonl`, not an active training split. A combined candidate is JSONL with a `split` value on every row. Candidate paths must end in `.candidate.jsonl`; the builder refuses to overwrite an existing non-default candidate or an active train, validation, or test file. Training outputs must be fresh, stay inside the model workspace, and remain separate from protected inputs. Do not delete or clear an output directory to get past preflight.

Generated artifacts include candidate datasets, run metadata, checkpoints, adapters, merged model files, GGUF files, and Ollama tags. They are distinct from source code and documentation. Data and model outputs are not created by running the test suite.

## Configuration

Environment variables are read when the process starts. Common settings include:

| Area | Variables |
| --- | --- |
| Input datasets | `HACKGRADER_DATASET_FILE`, `HACKGRADER_TRAIN_FILE`, `HACKGRADER_VAL_FILE`, `HACKGRADER_TEST_FILE` |
| Output locations | `HACKGRADER_OUTPUT_DIR`, `HACKGRADER_ADAPTER_DIR`, `HACKGRADER_MERGE_DIR`, `HACKGRADER_RUN_ID` |
| Model and training | `HACKGRADER_BASE_MODEL`, `HACKGRADER_HOURS`, `HACKGRADER_MAX_STEPS_PER_ROUND`, `HACKGRADER_MAX_GATE_CANDIDATES`, `HACKGRADER_BATCH`, `HACKGRADER_ACCUM`, `HACKGRADER_MAX_LEN` |
| Evaluation and gates | `HACKGRADER_CHECK_ROWS`, `HACKGRADER_MIN_TEST_ROWS`, `HACKGRADER_MIN_TEST_CLASS_ROWS`, `HACKGRADER_TEST_MIN_PRECISION`, `HACKGRADER_TEST_MIN_RECALL`, `HACKGRADER_TEST_MIN_SPECIFICITY`, `HACKGRADER_HOLDOUT_MAE_LIMIT`, `HACKGRADER_MIN_HOLDOUT_SPREAD`, `HACKGRADER_MIN_HOLDOUT_SEPARATION` |

Defaults and validation are in [`src/learning/config.py`](src/learning/config.py), [`src/learning/config_validation.py`](src/learning/config_validation.py), [`src/data/config.py`](src/data/config.py), and [`src/learning/training/trainer_config.py`](src/learning/training/trainer_config.py). The detailed candidate schema and score bands are in the [data reference](src/data/README.md).

## Code map

| Path | Responsibility |
| --- | --- |
| [`src/cli.py`](src/cli.py) | Dispatches `data`, `train`, `score`, and `deploy`. |
| [`src/data/`](src/data/README.md) | Candidate collection, rendering, provenance, validation, and splitting. |
| [`src/data/harvest/`](src/data/harvest/README.md) | Merged agent pull requests, per-file last-touch attribution, and the audited label manifest. |
| [`src/learning/`](src/learning/README.md) | Preflight, bounded training, holdout gates, scoring, and release. |
| [`src/model_paths.py`](src/model_paths.py) | Adds the shared root application packages to the import path. |
| `Modelfile` | Ollama template, system prompt, and default parameters used when a build is registered. |
| `data/deployed_model.json` | Release record written once registration is verified: version, tag, adapter hash, GGUF name, and deployment time. The grader reads this file to resolve its default tag. |
| `data/model_versions.json` | Sequential release history: each version, its run id, tag, adapter digest, GGUF path, and release time. |
| `data/` | Repository source lists, the harvest cohort, the label manifest, and generated candidate datasets. |
| `data/repos.txt` | Hand-written repository pool and the detector score recorded for each entry. |
| `data/repos.fresh.txt` | Separate repository pool used for source-disjoint splits and the detector measurement cited in the [data reference](src/data/README.md). |
| `data/cohort.jsonl`, `data/cohort.provenance.json` | Resumable harvest cohort and the record of the runs that produced it. |
| `data/repository_labels.jsonl` | Accepted and rejected repository labels the builder supervises from. |
| `scripts/` | Operational helpers: the harvest-to-candidate chain, the human-bias measurement, and the candidate-target check. |
| `reports/` | Result write-ups and build logs kept out of the root: human-bias measurement, labeled-build notes, and the recorded labeled build log. |
| `pyproject.toml`, `uv.lock` | The workspace's Python 3.12 environment and its pinned development tools. |
| `outputs/` | Run metadata, adapters, and checkpoints. Treat contents as generated artifacts. |
| `models/` | Merged weights and quantized GGUF files for each run. Generated artifacts, never sources. |
| `llama.cpp/` | Local conversion dependency, when present. |

For implementation boundaries, see the [pipeline reference](src/learning/pipeline/README.md), [preflight reference](src/learning/preflight/README.md), and [deployment reference](src/learning/deployment/README.md). The project is licensed under MIT.
