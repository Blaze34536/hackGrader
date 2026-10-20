# Training preflight

Preflight checks that the selected dataset and output destinations are safe before the GPU runtime loads. It verifies input structure, split provenance, distinct-prompt and class coverage, and output-path isolation. It can catch a bad dataset without starting training.

## Run preflight

From `model/`:

```bash
uv run src/cli.py train --preflight-only
uv run src/cli.py train --dataset data/dataset.candidate.jsonl --preflight-only
```

The first command uses the configured input. The second selects an explicit combined candidate. Preflight reports a runtime estimate, then exits; it does not load a model, train, score the independent test, merge weights, or deploy.

For fresh-repository test validation, use both flags:

```bash
uv run src/cli.py train \
  --dataset data/dataset.candidate.jsonl \
  --fresh-repository-test \
  --strict-gates \
  --preflight-only
```

The final test is then restricted to real-repository rows whose canonical repository URLs do not appear in train or validation. The remaining unique prompts must pass the configured row and score-class coverage checks. Non-repository rows are excluded; they do not count toward fresh-test minimums.

## Checks and path protections

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Re-exports the dataset, path, and row preflight entry points. |
| `train_inputs.py` | Selects the combined dataset or all three legacy split files. |
| `rows.py` | Reads JSONL and separates combined data by `split`. |
| `sizes.py` | Requires distinct prompts, class coverage, score bands, and boundary representation. |
| `dataset.py` | Checks source families, repository presence, lineage, and split leakage. |
| `paths.py` | Refuses unsafe candidate/output paths and overwrites. |

Output paths must be inside `model/`, fresh, and separate from active input files and protected artifacts. Preflight does not delete, repair, or overwrite a conflicting path; choose a different run ID or output location instead.

For builder-side checks, see the [candidate builder](../../data/builder/README.md). The [pipeline reference](../pipeline/README.md) describes where preflight sits in training and release.
