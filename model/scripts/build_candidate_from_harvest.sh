#!/usr/bin/env bash
# Harvest audited repository labels, then build the candidate dataset a training run reads.
#
# This is one command because the two steps share a contract: the builder refuses to read a
# real repository without a verified label, so a manifest that is missing, one-class, or
# unreadable must stop the chain before any dataset is written. The harvest is resumable —
# re-running this script after an interruption continues the cohort instead of restarting
# it — and the build writes `data/dataset.candidate.jsonl`, which `uv run src/cli.py
# train --candidates` reads.
#
# Usage, from anywhere:
#
#     ./scripts/build_candidate_from_harvest.sh
#
# Override the window or filenames with SINCE, UNTIL, MANIFEST, or REPOS.
set -euo pipefail
cd "$(dirname "$0")/.."

SINCE="${SINCE:-2025-01-01}"
UNTIL="${UNTIL:-2026-09-30}"
MANIFEST="${MANIFEST:-data/repository_labels.jsonl}"
REPOS="${REPOS:-data/candidate_repos.txt}"
# The documented cohort floor is 60 repositories, but 60 verified repositories produce
# roughly 900 evidence rows, which is short of the 1,600-row training floor once the
# label-evidence gate has done its work, and the AI-range quota needs repositories whose
# agent attribution covers a large share of the code. 220 over a 21-month window is the
# size that cleared every floor on 2026-10-18 while staying inside one hour of metered
# GitHub requests.
MAX_REPOS="${MAX_REPOS:-220}"

echo "[i] harvesting agent-attributed and human-class repositories ${SINCE}..${UNTIL}"
uv run src/cli.py harvest \
  --since "$SINCE" \
  --until "$UNTIL" \
  --per-agent-limit "${PER_AGENT_LIMIT:-80}" \
  --max-repositories "$MAX_REPOS" \
  --human-repositories data/human_repos.txt \
  --labels-out "$MANIFEST" \
  --pause 0.05

echo "[i] verifying ${MANIFEST} with the same fail-closed validator the builder uses"
uv run --project . python - "$MANIFEST" "$REPOS" <<'PY'
"""Refuse a manifest the builder cannot supervise both classes from."""

import sys

sys.path.insert(0, "src")

from data.labels import LABEL_AI, LABEL_HUMAN, load_repository_labels, measured_share
from data.validation import slug_exposes_label

manifest_path, repos_path = sys.argv[1], sys.argv[2]
manifest = load_repository_labels(manifest_path)
counts = manifest.counts()
measured = [
    share
    for entry in manifest.accepted.values()
    if (share := measured_share(entry)) is not None
]
print(
    f"[i] {len(manifest.accepted)} accepted label(s): "
    f"{counts[LABEL_AI]} ai, {counts[LABEL_HUMAN]} human, "
    f"{len(manifest.rejected)} rejected"
)
if measured:
    measured.sort()
    print(
        f"[i] measured attribution in {len(measured)} entr(ies): "
        f"median {measured[len(measured) // 2]:.0f}%, "
        f"range {measured[0]:.0f}-{measured[-1]:.0f}%"
    )
if not counts[LABEL_AI] or not counts[LABEL_HUMAN]:
    raise SystemExit(
        f"[!] A usable manifest needs both classes; found {counts[LABEL_AI]} ai and "
        f"{counts[LABEL_HUMAN]} human. The dataset was not built."
    )
urls = sorted(entry["url"] for entry in manifest.accepted.values())
# A repository called ``project-AI`` hands the model its label in the prompt, so the row
# contract refuses it. Dropping it here keeps a build from dying one row at a time.
kept = [url for url in urls if not slug_exposes_label(url)]
for url in sorted(set(urls) - set(kept)):
    print(f"[!] {url}: slug reveals the class; left out of the repository list")
with open(repos_path, "w", encoding="utf-8") as handle:
    handle.write(
        "# Repository URLs taken from the accepted label manifest; one per line.\n"
        + "\n".join(kept)
        + "\n"
    )
print(f"[i] wrote {len(kept)} repository URL(s) to {repos_path}")
PY

echo "[i] building the candidate dataset from verified labels"
uv run src/cli.py data -r "$REPOS" --repository-labels "$MANIFEST"
echo "[i] candidate ready: train with \`uv run src/cli.py train --candidates\`"
