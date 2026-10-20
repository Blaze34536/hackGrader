# Candidate data builder

The builder behind `uv run src/cli.py data` creates, audits, splits, and publishes a combined candidate dataset. It refuses to publish a candidate until row format, citations, token length, source lineage, split quotas, and output-path checks pass.

## Build a candidate

Run from `model/`:

```bash
# Harvest verified labels, derive the repository list, and build in one resumable chain.
./scripts/build_candidate_from_harvest.sh

uv run src/cli.py data -h
uv run src/cli.py data -r data/candidate_repos.txt --repository-labels data/repository_labels.jsonl
```

`--repos` is required unless `HACKGRADER_SNIPPET_ONLY=1`. It names a text file containing one GitHub repository URL per line. Repository data may come from the configured cache; otherwise the builder may need network access and GitHub API capacity. A `GITHUB_TOKEN` is optional and can raise rate limits.

`--repository-labels` takes a JSON Lines manifest of audited repository authorship labels, and it is **required** whenever `--repos` is given. Only repositories whose entries pass the fail-closed evidence standard in [`labels.py`](../labels.py) are read, and each verified label decides its target: the measured attributed share when the entry recorded one, 0 for a verified human-class repository, and the band midpoint only for an `ai` entry that recorded a class alone. A repository without an accepted label is excluded, a manifest covering only one class stops the build before any source loads, and `--repos` without a manifest stops instead of falling back to the detector's style reading. Build without repository rows by setting `HACKGRADER_SNIPPET_ONLY=1`.

The default output is `data/dataset.candidate.jsonl`, a combined JSONL with `train`, `validation`, and `test` tags. Use `--combined-out data/another.candidate.jsonl` to select a new combined output. Use `--separate-files --train-out ... --val-out ... --test-out ...` to write three distinct candidate files. Every candidate output must end in `.candidate.jsonl` and remain inside `model/`. Existing non-default candidates, active datasets, and unsafe output paths are protected from overwrite.

## Fresh-repository test option

Add `--fresh-repository-test` to ask the splitter for at least 50 distinct real-repository prompts in the test partition, including at least 10 prompts from each score class. The test prompts come from whole repository/source components reserved away from train and validation. You can set explicit floors with `--min-fresh-test-repository-prompts` and `--min-fresh-test-class-rows`; these options change requested minimums only and do not bypass other split or preflight gates.

The corresponding training flag `--fresh-repository-test` filters final-test rows to real repositories whose canonical URLs do not occur in train or validation. The holdout-coverage gate then checks the filtered rows. See [preflight](../../learning/preflight/README.md) and the [candidate data contract](../README.md).

## Build behavior and failure

The builder collects labeled snippets, real repository views, synthetic examples, and explicit-artifact examples. It removes conflicting prompts together with source-linked derivatives, merges safe duplicate prompts, caps target buckets, allocates whole lineage components, verifies all partitions, checks token length, and only then writes output. A quota or provenance failure stops the build and leaves existing candidate files untouched.

The default dataset contains heuristic teaching labels, not human-verified authorship judgments. Review candidate sources, labels, evidence, and split coverage before training.

## Package map

| File | Responsibility |
| --- | --- |
| `__init__.py` | Re-exports the builder entry point, candidate helpers, and split functions. |
| `candidates.py` | Collects candidate sources, renders rows, and validates the live data contract. |
| `splits.py` | Quarantines conflicts, deduplicates, caps targets, splits rows, and applies dataset-size gates. |
| `cli.py` | Parses options, protects output paths, and publishes combined or separate candidates. |
