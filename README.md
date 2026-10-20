# HackGrader

HackGrader reviews projects from Devpost galleries. It reads linked GitHub repositories, asks a local Ollama judge for a structured assessment, and saves the judge's scores with ranked evidence. Factual repository measurements may be included in the prompt as context; they never alter the judge's scores. The scores are screening evidence—not proof of authorship.

## Choose a path

- **Run the grader:** follow [Requirements and setup](#requirements-and-setup), then use [Application commands](#application-commands).
- **Understand the application:** start with the [source map](src/README.md), then see the [grading workflow](src/hackgrader/README.md).
- **Change or test the code:** use the [test guide](tests/README.md) and the package references linked from the source map.
- **Read the design reasoning:** the [design records](docs/README.md) hold the dated specs behind the score targets and the provenance work, including which direction superseded which.
- **Work on model data or training:** start with the [model workspace guide](model/README.md), then read its [data reference](model/src/data/README.md) and [learning workflow](model/src/learning/README.md).

## Requirements and setup

- Python 3.14 or newer and [uv](https://docs.astral.sh/uv/).
- [Ollama](https://ollama.com/) running with a grader model installed. Its tag is resolved from `HACKGRADER_MODEL`, then the release record at `model/data/deployed_model.json`, then the bare `hackgrader` tag.
- Chrome. The crawler drives a real browser session and resolves a matching driver automatically, through Selenium Manager on Windows and macOS and `webdriver-manager` on Linux.
- Network access to the Devpost gallery and to `api.github.com`. An optional `GITHUB_TOKEN` raises the GitHub API rate limit.
- A CUDA GPU, for the model workspace only: training and local inference are not practical on CPU.

Install the application and development dependencies from the repository root:

```bash
uv sync
```

Every run prints the build it resolved as `Grader model: <tag>` before grading starts. A committed `model/data/deployed_model.json` pins the build and its sequential version; if that tag is not installed locally, set `HACKGRADER_MODEL` to one that is. The grader never substitutes a different installed model on its own.

Model training uses a separate Python 3.12 environment. Install it only if you need the model workflow:

```bash
uv sync --project model --group dev
```

## Application commands

Run these commands from the repository root:

| Goal | Command |
| --- | --- |
| Show grader options | `uv run hackgrader -h` |
| Grade a Devpost gallery | `uv run hackgrader https://example.devpost.com` |
| Build a PDF from `scores.json` | `uv run hackgrader-pdf` |
| Choose PDF input and output | `uv run hackgrader-pdf scores.json report.pdf` |
| Run offline tests | `uv run pytest` (or `bash run_test.sh`) |
| Run live integration tests | `uv run pytest -m live -s` (or `bash run_test.sh live`) |
| Run every application test | `uv run pytest -m ''` (or `bash run_test.sh all`) |

A gallery run writes annotated project results to root `scores.json`. PDF generation reads saved results; it does not crawl repositories or call Ollama. Offline tests avoid Chrome and model services; plain `pytest` selects exactly those, which is why no file list is needed. Live tests need the relevant browser, network, and Ollama services; see the [test guide](tests/README.md) before running them.

## Run a gallery end to end

Two commands, in order, from the repository root:

```bash
uv run hackgrader https://example.devpost.com   # crawl, grade, and save results
uv run hackgrader-pdf                            # render the saved results as a PDF
```

The first command:

1. Crawls the gallery with Chrome and collects each project's GitHub link.
2. Closes the browser, then checks the Ollama server and the resolved model tag. A missing model stops the run here, before anything is graded.
3. Grades every unique repository: the judge returns one structured report per evidence batch, including AI evidence signs ranked strongest-to-weakest with cited explanations.
4. Rewrites root `scores.json` after each repository, then annotates it with the judge's scores, ranked evidence, deductions, and coverage. Automatic project flags are not generated.

Expect minutes per repository rather than seconds, because each repository is read over the GitHub API, split into batches that fit the model's token window, and answered one batch at a time. A gallery of a few dozen projects runs for hours. Results are saved as they complete, so an interrupted run keeps the projects it already finished — but a later run starts over and replaces `scores.json`, so copy it aside first when comparing builds.

Failures are reported instead of being scored over: a gallery with no usable GitHub repositories stops the workflow, a tag that is not installed is reported as missing, and a batch too dense for the token window returns status `413` for that repository only. A failed evaluation remains visible through its status and returned content; no automatic flags are added. Scores are screening evidence for human review, not proof of authorship.

## Model workflow

The model workspace builds a candidate dataset, fine-tunes an adapter, evaluates saved checkpoints, and registers a verified Ollama tag. It needs its own Python 3.12 environment (see above), a CUDA GPU for training and inference, and network access while building data.

A fresh clone has none of the generated inputs: the label manifest, cohort, candidate dataset, checkpoints, merged weights, and GGUF are all ignored by Git, and so is the `llama.cpp` converter. The [fresh-clone sequence](model/README.md#fresh-clone-build-the-data-then-train) lists the one-time setup, the resumable harvest that rebuilds the labels, and the preflight, train, and verification commands in order.

The first two commands run once per clone, from the repository root and then `model/`; the rest run from `model/`:

```bash
uv sync --project model --group dev        # Python 3.12 environment
cd model && git clone --depth 1 https://github.com/ggml-org/llama.cpp.git  # GGUF converter
export GITHUB_TOKEN=<token>                # harvest rate limit
bash scripts/build_candidate_from_harvest.sh  # harvest verified labels, then build data/dataset.candidate.jsonl
uv run src/cli.py train --preflight-only   # validate inputs and destinations; load nothing
uv run src/cli.py train                    # train, select on validation, gate on the reserved test
```

A passing run merges, converts, and registers its own model tag, so `train` is the release. To inspect afterwards, use `score` for validation results and `deploy --check` for the served artifact; `recover --run-id <id>` finishes an interrupted release without re-scoring the test.

Outputs land in `model/outputs/` (run metadata, checkpoints, adapters) and `model/models/` (merged weights and the quantized GGUF). The release record lives at `model/data/deployed_model.json`, and the version history at `model/data/model_versions.json`. Weights and run outputs are generated artifacts rather than source; the small release record is the exception, and its committed copy is what pins the grader's default tag.

Preflight checks inputs and destinations without training. Training uses validation to choose checkpoints and guide continuation; the independent test is reserved for a final gated evaluation and must not guide training. See [Model training workspace](model/README.md) for every command, configuration variable, data-safety rule, and release gate.

## Repository map

| Location | What you will find |
| --- | --- |
| [`src/`](src/README.md) | Application architecture and package references. |
| [`tests/`](tests/README.md) | Test commands, prerequisites, and suite boundaries. |
| [`docs/`](docs/README.md) | Dated design records, with their status and what each supersedes. |
| [`model/`](model/README.md) | Candidate data, training, scoring, and gated deployment. |
| [`LICENSE`](LICENSE) | MIT license. |

The code is organized around shared report and evidence contracts. The judge alone scores each project; the detector contributes factual prompt context only. Judge reports list distinct AI signs in strongest-to-weakest order with quotes, file names, and explanations, and no automatic project flags are generated. The application and model workspace use the same report format, so changes to prompts, citations, or score parsing should be reviewed across both workflows.
