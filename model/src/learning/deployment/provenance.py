"""Hash, locate, and name the immutable artifacts a deployment is built from.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import hashlib
import json
import os
import re
from pathlib import Path

from ..config import MERGE_DIR

MODEL_DIR = Path(__file__).resolve().parents[3]
GGUF_TYPE = os.environ.get("HACKGRADER_GGUF_TYPE", "q8_0")
TAG = os.environ.get("HACKGRADER_OLLAMA_TAG", "hackgrader")
MODELFILE = "Modelfile"
CONVERTER = "llama.cpp/convert_hf_to_gguf.py"
STAMP = "merge_stamp.json"
RELEASED_MODEL = "data/deployed_model.json"
_DIGEST_LENGTH = 16


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it into memory at once."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weight_time(path: str) -> float:
    """Return when a weight file was created or last written, whichever is newer.

    Unsloth's merge copies the cached 16-bit base shards into the merge directory,
    which carries their old modification times over, and then writes the merged
    tensors into those shards in place, which not every filesystem stamps. Creation
    time is what still moves when a merge produces a shard, so either timestamp
    counts as evidence that the merge ran.
    """
    status = os.stat(path)
    return max(status.st_mtime, status.st_ctime)


def blob_directory() -> Path | None:
    """Find the local Ollama content-addressed blob directory, if present."""
    candidates = [
        os.environ.get("OLLAMA_MODELS"),
        Path.home() / ".ollama" / "models",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        blobs = Path(candidate) / "blobs"
        if blobs.is_dir():
            return blobs
    return None


def served_gguf(gguf: Path) -> bool:
    """Check whether Ollama stores the exact GGUF file that was just built."""
    if not gguf.exists():
        print(
            f"[!] {gguf.name} is not on disk, so there is no build to compare "
            "against what ollama serves."
        )
        return False

    blobs = blob_directory()
    if blobs is None:
        print(
            "[!] Could not find ollama's model directory; refusing an unverifiable deploy."
        )
        return False

    size = gguf.stat().st_size
    if not any(blob.stat().st_size == size for blob in blobs.iterdir()):
        return False

    print(f"[i] Hashing {gguf.name} to compare it with what ollama serves...")
    return (blobs / f"sha256-{sha256_file(gguf)}").exists()


def adapter_stamp(merge_dir: str = MERGE_DIR) -> dict:
    """Load merged-checkpoint provenance metadata when it is available."""
    path = MODEL_DIR / merge_dir / STAMP
    if not path.exists():
        print(f"[!] {path} is missing, so the served adapter cannot be named.")
        return {}
    try:
        stamp = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"[!] Cannot read the adapter stamp at {path}: {exc}"
        ) from None
    if not isinstance(stamp, dict):
        raise SystemExit(f"[!] The adapter stamp at {path} is not a JSON object.")
    return stamp


def configured_versioned_name(configured_name: str, digest: str) -> Path:
    """Name the run-specific GGUF for a verified digest without touching the disk."""
    configured_path = Path(configured_name)
    return configured_path.with_name(
        f"{configured_path.stem}-{digest[:_DIGEST_LENGTH]}{configured_path.suffix or '.gguf'}"
    )


def versioned_gguf_name(configured_name: str, digest: str) -> str:
    """Derive an adapter-addressed GGUF path confined to the model workspace."""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise SystemExit(
            "[!] A full adapter SHA-256 stamp is required for safe deployment."
        )
    configured_path = Path(configured_name)
    if configured_path.is_absolute() or ".." in configured_path.parts:
        raise SystemExit(
            "[!] HACKGRADER_GGUF must be a relative path inside the model directory."
        )
    versioned = configured_versioned_name(configured_name, digest)
    absolute = (MODEL_DIR / versioned).resolve()
    try:
        if os.path.commonpath((str(MODEL_DIR.resolve()), str(absolute))) != str(
            MODEL_DIR.resolve()
        ):
            raise SystemExit(
                "[!] Versioned GGUF output must stay inside the model directory."
            )
    except ValueError:
        raise SystemExit(
            "[!] Versioned GGUF output must stay inside the model directory."
        ) from None
    if absolute.exists() or absolute.is_symlink():
        raise SystemExit(
            f"[!] Refusing to overwrite existing versioned GGUF {absolute}; "
            "inspect it or use a different model build."
        )
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return versioned.as_posix()


_versioned_gguf_name = versioned_gguf_name
