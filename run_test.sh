#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

mode="${1:-offline}"
if (($# > 0)); then
  shift
fi
if (($# > 0)); then
  printf 'Usage: bash run_test.sh [offline|live|all]\n' >&2
  exit 2
fi

run_app_tests() {
  if command -v uv >/dev/null 2>&1; then
    uv run --project "$root" --group dev pytest "$@"
  else
    python -m pytest "$@"
  fi
}

# Which tests run is configured once in pyproject.toml: testpaths keeps collection inside
# tests/, and addopts deselects the live ones. This wrapper therefore only selects a mode.
case "$mode" in
  offline)
    run_app_tests -q
    ;;
  live)
    run_app_tests -sv -m live
    ;;
  all)
    run_app_tests --override-ini "addopts=-q"
    ;;
  *)
    printf 'Unknown test mode: %s\nUsage: bash run_test.sh [offline|live|all]\n' "$mode" >&2
    exit 2
    ;;
esac
