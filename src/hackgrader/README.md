# Gallery-grading workflow

The `hackgrader` package is the command-line entry point and orchestration layer for grading a Devpost gallery. It discovers project links, extracts GitHub repository URLs, evaluates each repository, and saves annotated results to the project-root `scores.json`.

## Entry points and files

| File | Responsibility |
| --- | --- |
| `__init__.py` | Package marker with no re-exports; the installed command resolves to `hackgrader.cli:main`. |
| `cli.py` | Parses the required Devpost gallery URL for the installed `hackgrader` command. |
| `pipeline/__init__.py` | Exposes the stable `run_gallery()` and `individual_query()` imports. |
| `pipeline/workflow.py` | Manages browser cleanup, Ollama preflight, per-repository grading, and incremental persistence. |

See the [pipeline reference](pipeline/README.md) for the orchestration sequence, stable imports, and testing boundary.

## Run a gallery

From the repository root:

```bash
uv run hackgrader -h
uv run hackgrader https://example.devpost.com
```

The run needs network access, Chrome, readable GitHub repositories, and an available Ollama server/model. Set `HACKGRADER_MODEL` to an installed model tag and `OLLAMA_HOST` for a non-default Ollama server. An optional `GITHUB_TOKEN` increases GitHub API capacity. See the [grading reference](../grading/README.md) for model setup and the [crawling reference](../crawling/README.md) for browser behavior.

## Workflow contract

`run_gallery(url)` returns a mapping keyed by normalized `owner/repository` slug. It closes the browser after crawling and before grading, rejects galleries with no GitHub projects, checks grader readiness before processing repositories, and skips duplicate slugs. Each completed result is saved before the next repository is graded; if a later item fails, earlier results remain in `scores.json`.

Each raw result includes an HTTP-like integer `status` and human-readable `content`. Repository-backed results may also include prompt-context analysis, batch reports, and citation-verification details. `Reporting` stores the judge's parsed `ai_score` and `code_score`, summary, ranked evidence, deductions, and coverage. The judge alone scores projects. Prompt-context detector verdicts and findings are excluded; only factual measurements and coverage may accompany the prompt, and no automatic project flags are generated.

The report format and citations are defined by the shared [`contracts` package](../contracts/README.md). Unsupported model citations remain available for inspection but do not replace a score. Scores are screening evidence for human review—not proof of authorship or code quality.

## Output and privacy

The gallery command writes root `scores.json`; the PDF command can turn that saved file into a report. Repository evidence and result fields can contain copied source excerpts. Review output handling and avoid sharing scores or artifacts that include private or sensitive data.
