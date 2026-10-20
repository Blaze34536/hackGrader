# Gallery workflow internals

The `hackgrader.pipeline` package coordinates one gallery run. Its public entry points are re-exported from `hackgrader.pipeline`: use `run_gallery(url)` to run a gallery, or `individual_query(url, grader)` to evaluate one repository with an existing grader instance.

## What a gallery run does

1. Opens the gallery with the Selenium crawler and discovers project and GitHub links.
2. Closes the browser before grading starts, including when crawling raises an error.
3. Stops if the gallery has no usable GitHub repositories.
4. Checks that the configured Ollama server and model are ready.
5. Normalizes repository slugs, skips duplicate repositories, and grades each remaining repository.
6. Writes the accumulated results to `scores.json` after each repository, then annotates and sorts that file with `Reporting.generate_report()`. The printable PDF is a separate command; see the [PDF reference](../../pdf/README.md).

Each result contains an integer `status` and human-readable `content`. A handled repository-level error is saved as a result with status `500`, allowing later repositories to continue. Earlier saved results remain on disk if a later grading step fails.

## Module map

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Re-exports the stable `run_gallery` and `individual_query` imports. |
| `workflow.py` | Owns browser lifecycle, repository deduplication, grader readiness, result persistence, and report generation. |

The installed `hackgrader` command parses a gallery URL and calls `run_gallery`; see the [application workflow guide](../README.md). The resulting JSON is annotated by the [grading runtime](../../grading/README.md) and can be rendered by the [PDF command](../../pdf/README.md).

## Testing and boundaries

Use mocked crawlers, drivers, and graders for deterministic tests. The gallery integration tests require live browser/network access; the [test guide](../../../tests/README.md) lists the offline and live commands.

The package orchestrates crawling and grading but does not itself decide whether an authorship score is correct. Scores and evidence are screening information for human review, not proof of authorship.
