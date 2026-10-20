# Repository labels and the rebuilt candidate (2026-10-18)

What this replaces: the deployed adapter was trained on repository targets produced by the
detector's style reading, with no verified authorship label anywhere in the build. On twenty
repositories that ended before 2023 it answers a mean of **82.3** where the truth is about 0
(`human_bias.md`). The labels below are the first ones that come from a record instead.

## The manifest

`repository_labels.jsonl` — JSON Lines, one repository per line, written by
`uv run src/cli.py harvest` and validated by `data.labels` (the same fail-closed standard the
builder applies):

| | count |
| --- | --- |
| accepted `ai` labels | **220** |
| accepted `human` labels | **20** |
| rejected | **0** |

- **`ai`** entries carry the merged agent pull request, its merge commit, the agent app, and
  the measurement the label rests on: `attributed_lines`, `examined_lines`,
  `attributed_files`, `examined_files`. Median measured attribution is 3% (an agent pull
  request usually touches one corner of a mature repository), with a full range up to 100%.
- **`human`** entries carry the protocol that rules the class in: every commit predates
  2023-01-01, and no registered agent appears among the newest commits. The twenty
  repositories are the same dormant, human-written projects used in the bias measurement.
- Repositories the harvester could not attribute at file level (``no examined file was last
  touched by it``) and Devin repositories whose commits carry no agent identity are left out
  rather than softened.

## The candidate

`dataset.candidate.jsonl` — 2,100 rows, rebuilt from the manifest:

| split | rows | real-repository rows |
| --- | --- | --- |
| train | 1,500 | 916 |
| validation | 300 | 281 |
| test | 300 | 284 |

Real-repository targets are the measured attribution, not a band constant: mean **33.0**,
median **11**, range **0–100**, spread across every band (974 rows 0–39, 109 rows 40–59, 398
rows 60–100). That middle band is the part the old validation set was missing entirely.

Three code changes made this build possible, and each is a rule rather than a number:

1. **Real repositories require a manifest.** `--repos` without `--repository-labels` stops
   the build instead of scoring style (`builder/candidates.py`).
2. **The target is the measured share.** `provenance_target` reads `attributed_lines /
   examined_lines` (files as fallback), teaches 0 for a verified human-class repository, and
   drops an `ai` label whose own measurement is zero (`labels.py`).
3. **A verified label is cited from code measurements, not from style weights.** The style
   weights were fitted on snippets and a repository view without a README had no quotable
   feature, which dropped most verified batches for a reason the label never depended on
   (`VERIFIED_EVIDENCE_WEIGHTS` in `evidence.py`).

Also fixed while building: a citation of `for line in os.environ.get(...)` was read as a
cited file named `os.environ.get`, and a repository whose slug contains a class word (for
example `.../project-AI`) is now dropped from the pool rather than failing the row contract
one row at a time.

## Training tonight

From `model/`:

```bash
uv run src/cli.py train --candidates
```

Preflight has been run and passes: 2,100 rows, all four source families in every split, 1,363
distinct source groups, estimated **4.3 hours** for the full run inside the 20-hour cap.

Rebuilding the labels and the candidate from scratch (resumable, about an hour on a
`GITHUB_TOKEN`):

```bash
./scripts/build_candidate_from_harvest.sh
```

## What to check afterwards

Run the same bias measurement against the new tag, on repositories the training data never
saw. The twenty human-class repositories in this manifest are now training rows, so grading
them again would measure fit, not calibration:

```bash
uv run --project .. python scripts/measure_human_bias.py --cutoff 2023-01-01 \
  --out reports/human_bias_after.jsonl --exclude data/human_repos.txt
```

`--exclude` skips the repositories already in the training data, so what remains is unseen.
The old adapter means 82.3 on the original twenty. A checkpoint trained on measured targets
should land in single digits on unseen ones; if it does not, the remaining gap is
calibration rather than labels, and the release gate should keep it from shipping.
