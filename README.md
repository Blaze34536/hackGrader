<div align="center">

# HackGrader

**Reads hackathon repositories and returns a structured review of how much of the code looks AI-written, with the evidence attached.**

![Python](https://img.shields.io/badge/python-3.14%2B-3776AB?logo=python&logoColor=white)
![Model env](https://img.shields.io/badge/model%20env-3.12-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-2ea44f)

</div>

HackGrader crawls a Devpost gallery, finds each project's GitHub repository, and asks a local Ollama model to review it. Every project comes back with an AI-involvement score, a code-quality score, and ranked evidence quoted from the files that were actually read. Factual measurements are included as context; they never change the judge's scores.

Scores support human review. They are not proof of who wrote a project.

## Quick start

You need Python 3.14+ and [uv](https://docs.astral.sh/uv/), a running [Ollama](https://ollama.com/) with a grader model installed, Chrome, and network access to Devpost and GitHub.

```bash
uv sync                                          # install the application
uv run hackgrader https://example.devpost.com    # crawl, grade, write scores.json
uv run hackgrader-pdf                            # render scores.json as a PDF
```

Every run prints the model it resolved as `Grader model: <tag>` before grading. If that tag is not installed, set `HACKGRADER_MODEL` to one that is; the grader never swaps in a different model on its own.

## What a run does

- **Crawl** the gallery in Chrome and collect each project's repository link.
- **Read** every repository in batches that fit the model's window, then ask for one structured report per batch.
- **Save** `scores.json` after each repository, annotating it with scores, ranked citations, deductions, and coverage.

Repositories take minutes rather than seconds, so a gallery of a few dozen projects runs for hours. An interrupted run keeps what it finished, but the next run replaces `scores.json`, so copy it aside when comparing builds.

## Commands

| Goal | Command |
| --- | --- |
| Show options | `uv run hackgrader -h` |
| Grade a gallery | `uv run hackgrader <gallery-url>` |
| Build a PDF | `uv run hackgrader-pdf [scores.json] [report.pdf]` |
| Offline tests | `uv run pytest` |
| Live tests | `uv run pytest -m live -s` |

## Going deeper

| Read this | For |
| --- | --- |
| [Source map](src/README.md) | How the packages fit together, with a reference for each. |
| [Model workspace](model/README.md) | Building training data, fine-tuning, scoring, and the gated release. |
| [Test guide](tests/README.md) | Suite boundaries, prerequisites, and what live tests need. |
| [Design records](docs/README.md) | The dated reasoning behind the scoring targets and provenance work. |

The model workspace is the only part of the project that trains or deploys a model, and it is a separate Python 3.12 environment with its own CUDA, data, and release requirements.

```bash
uv sync --project model --group dev
```

Start at its [fresh-clone sequence](model/README.md#fresh-clone-build-the-data-then-train), which covers the one-time setup, the resumable harvest, and the preflight and training commands in order.

Application and model workspace share the report contract in `src/contracts`, so prompt, citation, or parsing changes should be reviewed in both.

## License

MIT. See [LICENSE](LICENSE).
