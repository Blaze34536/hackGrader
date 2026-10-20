"""Assign one sequential version to each verified model release.

The ledger is committed beside the release pointer so a version always names a build that
passed its gates: a number is written only after the GGUF is verified as served, a retry of
the same adapter digest reuses its version instead of advancing the sequence, and a preview
of the next number never writes anything.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .provenance import MODEL_DIR

LEDGER = "data/model_versions.json"


def ledger_path(path: str | Path | None = None) -> Path:
    """Return the ledger file, defaulting to the committed release history."""
    return Path(path) if path is not None else MODEL_DIR / LEDGER


def load_ledger(path: str | Path | None = None) -> dict[str, Any]:
    """Read the release history, refusing a file that cannot name its versions."""
    target = ledger_path(path)
    if not target.exists():
        return {"latest": 0, "models": []}
    try:
        ledger = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[!] Cannot read the model version ledger {target}: {exc}") from None
    if not isinstance(ledger, dict) or not isinstance(ledger.get("models"), list):
        raise SystemExit(
            f"[!] {target} must be a JSON object with a models list; no version was assigned."
        )
    entries = ledger["models"]
    versions = [
        entry.get("version") for entry in entries if isinstance(entry, dict)
    ]
    latest = ledger.get("latest")
    if (
        not isinstance(latest, int)
        or isinstance(latest, bool)
        or latest < 0
        or sorted(versions) != list(range(1, latest + 1))
        or len(entries) != latest
    ):
        raise SystemExit(
            f"[!] {target} has a latest value that does not match its recorded "
            "versions; inspect the ledger before releasing another model."
        )
    return ledger


def version_for_digest(ledger: dict[str, Any], digest: str) -> int | None:
    """Return the version already recorded for an adapter digest, when there is one."""
    for entry in ledger["models"]:
        if entry.get("digest") == digest:
            return int(entry["version"])
    return None


def preview_version(digest: str, path: str | Path | None = None) -> int:
    """Return the version a release would receive without writing the ledger."""
    ledger = load_ledger(path)
    recorded = version_for_digest(ledger, digest)
    return recorded if recorded is not None else int(ledger["latest"]) + 1


def version_tag(base: str, version: int) -> str:
    """Return the human-readable Ollama tag for one released version."""
    return f"{base}:v{version}"


def assign_version(
    digest: str,
    *,
    run_id: str,
    tag: str,
    gguf: str,
    released_at: str,
    expected_version: int | None = None,
    path: str | Path | None = None,
) -> int:
    """Record a verified release and return its version, reusing a known digest.

    ``expected_version`` is the number the deployment prepared and registered. If another
    process recorded a release in the meantime, the allocation stops instead of writing a
    ledger entry whose tag does not match what Ollama was asked to create.
    """
    ledger = load_ledger(path)
    recorded = version_for_digest(ledger, digest)
    if recorded is not None:
        return recorded
    version = int(ledger["latest"]) + 1
    if expected_version is not None and expected_version != version:
        raise SystemExit(
            f"[!] Another deployment recorded v{version - 1} while this one was "
            f"preparing v{expected_version}; no ledger entry was written. Rerun the "
            "release so the tag and the ledger agree."
        )
    ledger["models"].append(
        {
            "version": version,
            "run_id": run_id,
            "tag": tag,
            "digest": digest,
            "gguf": gguf,
            "released_at": released_at,
        }
    )
    ledger["latest"] = version
    _write_ledger(ledger, path)
    return version


def _write_ledger(ledger: dict[str, Any], path: str | Path | None = None) -> None:
    """Replace the ledger atomically so an interrupted release cannot truncate history."""
    target = ledger_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, target)


__all__ = [
    "LEDGER",
    "assign_version",
    "ledger_path",
    "load_ledger",
    "preview_version",
    "version_for_digest",
    "version_tag",
]
