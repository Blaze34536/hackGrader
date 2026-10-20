# Model source map

The model workspace is a separate Python 3.12 project. Its command entry point is `cli.py`; that dispatcher adds the repository-root `src/` packages so model and application code share the same report contract and detector implementation.

## Main packages

| Path | Responsibility | Reference |
| --- | --- | --- |
| `cli.py` | Dispatches `data`, `train`, `score`, and `deploy`. | [Workspace command guide](../README.md) |
| `model_paths.py` | Adds the root application source tree to imports. | Used by the workspace entry point. |
| `data/` | Builds, validates, and splits reviewable candidate datasets. | [Candidate-data reference](data/README.md) |
| `learning/` | Preflight, training, holdout evaluation, checkpoint scoring, and release. | [Learning workflow](learning/README.md) |

## Running commands

Run the CLI with `model/` as the working directory:

```bash
uv run src/cli.py -h
uv run src/cli.py data -h
uv run src/cli.py train --preflight-only
```

The model-side learning and training-data suites are no longer part of the repository, so changes here are verified through the application contracts under `tests/` instead. See the [model workspace README](../README.md) for installation and release safety.
