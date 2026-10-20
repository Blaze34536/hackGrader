# Grading runtime

The `grading/` package reads bounded repository evidence, asks an Ollama judge for the shared report format, validates its response, and returns the report for human review. The judge alone scores each project. Factual detector measurements may be included as prompt context, but are explicitly excluded as verdicts, evidence, or score inputs.

## Runtime interface

Use `grading.runtime.Model` to query repository questions containing a GitHub URL. `query()` and `query_with_retry()` return a mapping with an integer HTTP-like `status` and human-readable `content`. Successful repository-backed results can also include prompt-context `analysis`, batch reports, and citation-verification details.

The grader requests a three-part report defined by the shared [`contracts` package](../contracts/README.md): AI-involvement score with judge-ranked evidence, code-quality score with deductions, and a concise summary. AI evidence entries are ranked strongest-to-weakest; each names a distinct sign, quotes the observed text, cites its file, and explains how directly that sign supports the score. Each score is asked to carry a `- Why this score:` line naming the code that sets it and what would move it, so the two numbers arrive with a stated reason instead of standing alone. The judge is instructed not to treat generic style traits as proof of AI authorship, and to weigh the code over the README or any other prose: a README is the project's own description of itself, so nothing in it can raise either score on its own. The prompt asks for those justifications; the parser does not require them, so a report with a sound score is not rejected over a missing sentence.

Repository evidence is bounded and may be sent in several batches. Each citation is checked against the exact evidence view supplied for that batch: quotations must appear in that view, any named file must be one the view reproduces, and a quote taken from bookkeeping (a header, coverage count, or excerpt marker) is not repository evidence. Unsupported citations remain available in `unsupported_citations`; they do not change the judge's score. This is an audit detail, not a project flag.

`ai_score` and `code_score` are taken only from the judge's report. Prompt measurements include factual counts and coverage only; detector verdict scores and findings are not passed to the judge or saved as evaluation results. Results contain no automatic project flags or report findings. A failed request remains visible through its non-200 `status` and `content`; an invalid successful response has null parsed scores and its original report text remains available.

A batch that produces no valid report does not discard the repository if other batches produced valid reports. Scores, evidence, and deductions aggregate from valid batches; failed outputs remain in `failed_batches` and `batch_reports`. A repository fails outright only when no batch produced a valid report.

Before answering, each batch prompt is counted with the served model's tokenizer and must fit the model's trained token window. The prompt budget includes the system prompt, measurements, question, and repository text. When a prompt does not fit, the view is re-cut by the measured overflow and counted again, up to a bounded number of attempts. A batch that cannot be fitted returns status `413` without a generation.

`Reporting` parses and persists `ai_score`, `code_score`, `summary`, `metrics`, judge-ranked `evidence`, `deductions`, `coverage`, and citation details. Old flags, report findings, detector verdicts, and detector findings are removed when annotating an older result; factual prompt measurements remain available.

## Ollama configuration

The default Ollama host is `http://127.0.0.1:11434`; the default model tag is `hackgrader`.

| Variable | Purpose |
| --- | --- |
| `OLLAMA_HOST` | Ollama server address; defaults to the local host above. |
| `HACKGRADER_MODEL` | Installed model tag to query. The configured model is not silently replaced by another installed model. |
| `HACKGRADER_OLLAMA_NUM_CTX` | Context length requested for chat; defaults to `4096` and is never set below the trained window. |
| `HACKGRADER_MAX_LEN` | Token window the model was trained on; defaults to `3072`. Training and serving read this same variable. |
| `HACKGRADER_NUM_PREDICT` | Response tokens reserved inside that window; defaults to `800`. The prompt budget is the window minus this reserve. |
| `GITHUB_TOKEN` | Optional token for higher GitHub repository-read limits. |

Start Ollama and install the selected model before grading. The client checks server readiness and verifies that the configured tag is installed. Set a non-default host with `OLLAMA_HOST`; install a different model and set `HACKGRADER_MODEL` to its exact tag. Gallery runs perform a readiness preflight before grading projects.

### How the model tag is chosen

Resolution stops at the first source that yields a tag, so the order matters:

1. `HACKGRADER_MODEL`, when the environment sets it.
2. The release record at `model/data/deployed_model.json`, read by `released_model_tag()`. The deployment pipeline writes this file only after confirming that Ollama serves the exact GGUF it names, so a committed record pins the build the results came from.
3. The bare default tag `hackgrader` when the record is absent or unreadable.

The resolved tag is printed at startup as `Grader model: <tag>`. If that tag is not installed at the configured host, the run reports the missing model instead of substituting another installed tag.

The default chat options use temperature `0.2`, the configured response reserve, and a stable per-batch seed. Network, repository, and model failures appear as non-200 statuses or workflow errors; inspect `status` and `content` rather than trusting a numeric score on its own.

## Module map

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Re-exports `Model` and `Reporting`. |
| `ollama.py` | Configures the client, checks service/model readiness, and sends chat requests. |
| `runtime.py` | Reads evidence and coordinates model requests. |
| `evaluation.py` | Validates each answer and combines batch results. |
| `retry.py` | Retries transient model failures. |
| `window.py` | Fits each evidence view inside the model's trained token window. |
| `coverage.py` | Summarizes how much repository content was shown. |
| `reporting.py` | Parses judge-owned scores and evidence before saving results. |

A bounded dump may omit repository files or truncate excerpts, and batch summaries can miss context across views. Scores remain evidence for human review, not a factual determination of authorship.