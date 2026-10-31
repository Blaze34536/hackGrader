#!/usr/bin/env bash
# Download the weight parts from the GitHub release, verify, and unpack into model/ (models/ and outputs/).
set -euo pipefail
cd "$(dirname "$0")"
base=https://github.com/Blaze34536/hackGrader/releases/download/model-weights-2026-10-31
for f in $(awk '{print $2}' SHA256SUMS); do [ -f "$f" ] || curl -fL -o "$f" "$base/$f"; done
sha256sum -c SHA256SUMS
cat hackgrader_weights.tar.part-* | tar -xf - -C ../model
echo "restored into model/models and model/outputs"
