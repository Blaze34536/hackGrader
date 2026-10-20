# Test suite reference

Run commands from the repository root. The application uses the root Python 3.14+ environment.

## Choose a test mode

Which tests run is configured once in [`pyproject.toml`](../pyproject.toml): `testpaths` keeps collection inside `tests/` instead of the vendored `model/llama.cpp` test tree, and `addopts` deselects the live tests. So plain pytest is the offline suite, and a marker selects the live ones.

| Mode | Command | What it runs |
| --- | --- | --- |
| Offline (default) | `uv run pytest` | Deterministic application contracts and evidence-window checks. No Chrome or Ollama request. |
| Live | `uv run pytest -m live -s` | Gallery crawling, gallery grading, and live Ollama scoring. Needs the browser, network, GitHub, and Ollama prerequisites. |
| All | `uv run pytest -m ''` | Every test in `tests/`. |
| One file or class | `uv run pytest tests/test_scoring.py` | Only that selection, with the live tests still deselected. Add `-m live` to include them, for example `uv run pytest tests/test_scoring.py -m live`. |

`bash run_test.sh [offline|live|all]` and `./run_test.ps1 -Mode <mode>` run the same three modes for a shell that prefers a wrapper; they stop when the command fails.

Live scoring writes where you can read it. `TestScoring` prints a summary table — repository, judge AI, judge code, and the number of files shown — and saves each report with prompt-context coverage to `build/scoring/last-run.json` (ignored by version control).

pytest captures a passing test's output unless `-s` is passed, so the score table is repeated in the session summary under `scoring results` whether or not output was captured, and the table is printed again when a repository fails partway through. Both read the artifact, and only an artifact written during the running session is shown, so an offline run never reports an earlier run's scores. Every repository is saved as it finishes, so a failure on the last one keeps the scores of the first ones. If no model is reachable the live test fails immediately with the reason, rather than appearing to score nothing. `TestPipeline` writes its annotated scores to a temporary file that it deletes, so use `build/scoring/last-run.json` or a real `hackgrader` run for anything you want to keep.

Live tests may contact external services and should be run only when that behavior is intended.

## Focused commands

```bash
uv run pytest tests/test_scoring.py -m live   # just the live scoring test
uv run pytest -k citations                    # tests whose name matches
uv run ruff check src tests                   # lint the application and its tests
```

Add `--project . --group dev` to a command only when the environment is not already active; the wrappers add it for you. For additional examples and suite boundaries, see the [test fixtures](fixtures/README.md) and [test tools](tools/README.md).

## Test map

| Path | Purpose |
| --- | --- |
| `test_detector.py` | Legacy heuristic regressions plus report parsing and citation contracts. |
| `test_report.py` | Printable report layout: score columns, panels, truncated failures, and the rectangles the pages actually draw. |
| `test_runtime_contracts.py` | Compatibility of shared report/evidence contracts with the model runtime, plus repository evidence selection and the truncated-tree fallback. |
| `test_scoring.py` | Offline evidence-window checks and a live Ollama scoring integration case. |
| `test_crawling.py` | Devpost gallery browsing and project discovery. |
| `test_pipeline.py` | Gallery grading and result persistence. |
| `test_window.py` | Prompt assembly, window overflow, and evidence refitting. |
| `fixtures/` | Fixed, offline repository examples. |
| `tools/` | Timing and hardware diagnostics for live checks. |

Offline contract tests verify code behavior and sequencing; they do not establish deployed-model accuracy. Keep credentials and private repository data out of fixtures.
