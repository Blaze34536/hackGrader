"""Convert verified merged weights to immutable GGUF artifacts and Ollama tags.

``run`` lives here rather than in a submodule so that the module-level names it uses
(``convert``, ``register``, ``GGUF``) stay patchable at ``learning.deployment``, which is
how the offline pipeline tests intercept the shell-outs.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import json
import os
import time

from ..config import GGUF, MERGE_DIR, RUN_ID
from .commands import convert, register
from .modelfile import _versioned_modelfile
from .provenance import (
    _DIGEST_LENGTH,
    CONVERTER,
    GGUF_TYPE,
    MODEL_DIR,
    MODELFILE,
    RELEASED_MODEL,
    STAMP,
    TAG,
    adapter_stamp,
    blob_directory,
    configured_versioned_name,
    served_gguf,
    sha256_file,
    versioned_gguf_name,
)
from .provenance import versioned_gguf_name as _versioned_gguf_name
from .versioning import assign_version, preview_version
from .versioning import version_tag as _version_tag

__all__ = [
    "CONVERTER",
    "GGUF_TYPE",
    "MODELFILE",
    "MODEL_DIR",
    "RELEASED_MODEL",
    "STAMP",
    "TAG",
    "adapter_stamp",
    "blob_directory",
    "configured_versioned_name",
    "convert",
    "main",
    "register",
    "run",
    "served_gguf",
    "sha256_file",
    "versioned_gguf_name",
]


def run(
    dry_run: bool = False,
    convert_only: bool = False,
    *,
    merge_dir: str = MERGE_DIR,
    modelfile: str = MODELFILE,
    tag: str | None = None,
) -> bool:
    """Build and register an immutable model without replacing the default tag."""
    stamp = adapter_stamp(merge_dir)
    digest = str(stamp.get("adapter_sha256", ""))
    outfile_name = _versioned_gguf_name(GGUF, digest)
    model_tag = tag or TAG
    versioned = tag is None
    if versioned:
        version = preview_version(digest)
        release_tag = _version_tag(model_tag, version)
    else:
        version = None
        release_tag = f"{model_tag}:{digest[:_DIGEST_LENGTH]}"

    convert(merge_dir, outfile_name, GGUF_TYPE, dry_run)
    if convert_only:
        return True

    with _versioned_modelfile(outfile_name, modelfile) as generated_modelfile:
        register(str(generated_modelfile), release_tag, dry_run)

    if dry_run:
        if versioned:
            print(f"[i] dry run: would record {release_tag} as version {version}.")
        return True

    outfile = MODEL_DIR / outfile_name
    if served_gguf(outfile):
        if versioned:
            version = assign_version(
                digest,
                run_id=RUN_ID,
                tag=release_tag,
                gguf=outfile_name,
                released_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                expected_version=version,
            )
        record_release(release_tag, digest, outfile_name, version=version)
        print(f"[i] ollama registered {release_tag} from {outfile.name}.")
        if versioned:
            print(f"[i] recorded {release_tag} as model version v{version}.")
        print(f"[i] The grader adopts this build by default; override it with "
              f"HACKGRADER_MODEL={release_tag}.")
        return True

    print(
        f"[!] No ollama blob matches {outfile.name}. The versioned tag {release_tag} "
        "could not be verified; the existing default model tag was left unchanged."
    )
    return False


def record_release(
    tag: str, digest: str, gguf_name: str, *, version: int | None = None
) -> None:
    """Record the verified release tag where the grader resolves its default."""
    record = {
        "tag": tag,
        "adapter_sha256": digest,
        "gguf": gguf_name,
        "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if version is not None:
        record["version"] = version
    record_path = MODEL_DIR / RELEASED_MODEL
    os.makedirs(record_path.parent, exist_ok=True)
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)


def main(argv: list[str] | None = None) -> int:
    """Parse deployment options and return a shell-friendly exit status."""
    from .cli import main as _main

    return _main(argv)
