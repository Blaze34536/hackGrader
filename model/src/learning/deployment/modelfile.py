"""Generate the run-specific Ollama Modelfile that names one immutable GGUF.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .provenance import MODEL_DIR, MODELFILE


@contextmanager
def _versioned_modelfile(
    gguf_name: str,
    source_modelfile: str = MODELFILE,
    *,
    temp_root: str | Path = MODEL_DIR,
    model_dir: str | Path = MODEL_DIR,
):
    """Yield a temporary Modelfile referencing the immutable run-specific GGUF."""
    workspace = Path(model_dir).resolve()
    temporary_root = Path(temp_root).resolve()
    try:
        if os.path.commonpath((str(workspace), str(temporary_root))) != str(workspace):
            raise SystemExit(
                "[!] Temporary Modelfile must stay inside the model workspace."
            )
    except ValueError:
        raise SystemExit(
            "[!] Temporary Modelfile must stay inside the model workspace."
        ) from None
    source_path = Path(source_modelfile)
    if not source_path.is_absolute():
        source_path = workspace / source_path
    try:
        contents = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(
            f"[!] Cannot read the Modelfile at {source_path}: {exc}"
        ) from None
    lines = contents.splitlines()
    if not lines or not lines[0].startswith("FROM "):
        raise SystemExit(f"[!] {source_path} must begin with a FROM instruction.")
    with tempfile.TemporaryDirectory(
        prefix=".hackgrader-deploy-", dir=temp_root
    ) as temp:
        path = Path(temp) / MODELFILE
        temp_path = Path(temp).resolve()
        gguf_path = Path(gguf_name)
        gguf_absolute = gguf_path if gguf_path.is_absolute() else workspace / gguf_path
        gguf_absolute = gguf_absolute.resolve()
        try:
            if os.path.commonpath((str(workspace), str(gguf_absolute))) != str(
                workspace
            ):
                raise SystemExit(
                    "[!] Modelfile GGUF must stay inside the model workspace."
                )
        except ValueError:
            raise SystemExit(
                "[!] Modelfile GGUF must stay inside the model workspace."
            ) from None
        relative_gguf = os.path.relpath(gguf_absolute, temp_path).replace("\\", "/")
        try:
            lines[0] = f"FROM ./{relative_gguf}"
        except IndexError:
            raise SystemExit(
                f"[!] {source_path} must begin with a FROM instruction."
            ) from None
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        yield path
