"""Command-line entry point for building and registering the served model.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import argparse
import re
from pathlib import Path

from ..config import GGUF, MERGE_DIR
from .provenance import (
    MODEL_DIR,
    MODELFILE,
    adapter_stamp,
    configured_versioned_name,
    served_gguf,
)
from .versioning import load_ledger, version_for_digest


def _check(merge_dir: str) -> int:
    """Verify the currently configured GGUF against a run-specific build."""
    stamp = adapter_stamp(merge_dir)
    digest = str(stamp.get("adapter_sha256", ""))
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        print(
            "[!] A verified adapter hash is unavailable; cannot check a run-specific model."
        )
        return 1
    configured_path = Path(GGUF)
    if configured_path.is_absolute() or ".." in configured_path.parts:
        print("[!] HACKGRADER_GGUF must be inside the model directory.")
        return 1
    recorded = version_for_digest(load_ledger(), digest)
    if recorded is None:
        print("[i] this adapter has no recorded version yet.")
    else:
        print(f"[i] recorded version: v{recorded}.")
    versioned = configured_versioned_name(GGUF, digest)
    return 0 if served_gguf(MODEL_DIR / versioned) else 1


def main(argv: list[str] | None = None) -> int:
    """Parse deployment options and return a shell-friendly exit status."""
    from . import run

    parser = argparse.ArgumentParser(description="Deploy the merged HackGrader model.")
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="print the commands only"
    )
    parser.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="only verify the currently configured GGUF",
    )
    parser.add_argument(
        "-x", "--convert-only", action="store_true", help="skip Ollama registration"
    )
    parser.add_argument(
        "-m", "--merge-dir", default=MERGE_DIR, help="merged checkpoint directory"
    )
    parser.add_argument(
        "-f", "--modelfile", default=MODELFILE, help="Ollama Modelfile template"
    )
    args = parser.parse_args(argv)

    if args.check:
        return _check(args.merge_dir)

    return (
        0
        if run(
            dry_run=args.dry_run,
            convert_only=args.convert_only,
            merge_dir=args.merge_dir,
            modelfile=args.modelfile,
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
