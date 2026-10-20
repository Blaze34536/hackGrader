"""Run the external GGUF converter and the Ollama registration command.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import subprocess
import sys

from .provenance import CONVERTER, MODEL_DIR


def convert(merge_dir: str, outfile: str, gguf_type: str, dry_run: bool) -> None:
    """Convert merged Hugging Face weights to GGUF, or only print the command."""
    command = [
        sys.executable,
        CONVERTER,
        merge_dir,
        "--outfile",
        outfile,
        "--outtype",
        gguf_type,
    ]
    print("[i] " + " ".join(command))
    if dry_run:
        return
    try:
        subprocess.run(command, cwd=MODEL_DIR, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            f"[!] GGUF conversion failed ({exc}). The merged model is still at "
            f"{MODEL_DIR / merge_dir}; convert it by hand or use a new adapter build."
        ) from None


def register(modelfile: str, tag: str, dry_run: bool) -> None:
    """Register a versioned Ollama tag from a run-specific Modelfile."""
    command = ["ollama", "create", tag, "-f", modelfile]
    print("[i] " + " ".join(command))
    if dry_run:
        return
    try:
        subprocess.run(command, cwd=MODEL_DIR, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(
            f"[!] Could not register '{tag}' with ollama ({exc}). The versioned GGUF is "
            "preserved; retry deployment with a fresh tag or inspect Ollama."
        ) from None
