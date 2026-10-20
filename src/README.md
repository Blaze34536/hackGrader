# Application source

The application is organized as small packages under `src/`. The installed `hackgrader` command starts at `hackgrader.cli`; `hackgrader.pipeline.run_gallery()` coordinates a full gallery run. The separate `hackgrader-pdf` command starts at `pdf.cli` and formats results that have already been saved.

## Find the right package

| Package | Responsibility | Reference |
| --- | --- | --- |
| `hackgrader/` | Connects gallery crawling, grading, and result persistence. | [Workflow and result format](hackgrader/README.md) |
| `crawling/` | Opens Devpost pages and discovers project and repository links. | [Browser lifecycle](crawling/README.md) |
| `repositories/` | Reads public GitHub files and builds bounded evidence dumps. | [Repository access](repositories/README.md) |
| `detection/` | Measures authorship and code-quality signals deterministically. | [Signals and limitations](detection/README.md) |
| `contracts/` | Defines the shared report, citation, prompt, and repository-dump formats. | [Report and evidence contract](contracts/README.md) |
| `grading/` | Sends evidence to Ollama, validates reports, and annotates results. | [Runtime and output fields](grading/README.md) |
| `pdf/` | Renders saved scores as printable reports. | [PDF command and behavior](pdf/README.md) |

The application and model-training workspace both use `contracts`. When changing prompts, report structure, evidence markers, or citation rules, check both consumers and their contract tests.

## Commands

Run from the repository root:

```bash
uv run hackgrader -h
uv run hackgrader https://example.devpost.com
uv run hackgrader-pdf -h
uv run hackgrader-pdf scores.json report.pdf
```

A gallery run needs network access, Chrome, GitHub access, and an Ollama server with the selected model available. Configure `HACKGRADER_MODEL` for the Ollama model tag and `OLLAMA_HOST` for a non-default server; an optional `GITHUB_TOKEN` provides higher GitHub API limits. Gallery results are written to root `scores.json`. PDF generation reads that file by default and does not call the crawler, GitHub, or Ollama.

## Boundaries and limitations

Repository reads are bounded: only selected files and excerpts fit into each model request. Results should describe evidence actually shown to the model, and citations are checked against that evidence. The deterministic detector and model are fallible; scores are useful for screening and review, not proof of who wrote a project. See the [root README](../README.md) for setup and the [test guide](../tests/README.md) for offline and live checks.
