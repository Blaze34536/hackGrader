"""Refuse candidate and output paths that overwrite anything worth keeping.

Nothing here deletes or clears a directory. A run that cannot write somewhere fresh
stops instead, so a mistyped run id can never destroy the artifacts it was compared to.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import os

from ..config import COMBINED_FILE, TEST_FILE, TRAIN_FILE, VAL_FILE

_SOURCE_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def resolve_candidate_paths(
    *candidate_paths: str,
    active_paths: tuple[str, ...],
    root: str = ".",
    allow_existing: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Validate candidate filenames, allowing explicit in-root replacements only."""
    if not candidate_paths:
        raise ValueError("at least one candidate path is required")
    root_path = os.path.realpath(root)
    active = {
        os.path.normcase(
            os.path.realpath(
                path if os.path.isabs(path) else os.path.join(root_path, path)
            )
        )
        for path in active_paths
    }
    allowed_existing = {
        os.path.normcase(
            os.path.realpath(
                path if os.path.isabs(path) else os.path.join(root_path, path)
            )
        )
        for path in allow_existing
    }
    resolved_paths: list[str] = []
    for candidate in candidate_paths:
        absolute = os.path.abspath(
            candidate
            if os.path.isabs(candidate)
            else os.path.join(root_path, candidate)
        )
        resolved = os.path.realpath(absolute)
        try:
            inside_root = os.path.commonpath((root_path, resolved)) == root_path
        except ValueError:
            inside_root = False
        if not inside_root:
            raise SystemExit(
                "[!] Candidate files must stay inside the model directory."
            )
        if os.path.normcase(resolved) in active:
            raise SystemExit(
                "[!] Candidate paths may not replace an active training dataset."
            )
        if not os.path.basename(resolved).endswith(".candidate.jsonl"):
            raise SystemExit(
                "[!] Candidate outputs must use a *.candidate.jsonl name so they "
                "cannot be mistaken for active training inputs."
            )
        if (
            os.path.lexists(absolute)
            and os.path.normcase(resolved) not in allowed_existing
        ):
            raise SystemExit(
                f"[!] Refusing to overwrite an existing candidate dataset: {absolute}"
            )
        resolved_paths.append(resolved)

    normalized = [os.path.normcase(path) for path in resolved_paths]
    if len(normalized) != len(set(normalized)):
        raise SystemExit("[!] Candidate dataset paths must differ.")
    return tuple(resolved_paths)


def require_fresh_dir(path: str, what: str, *, root: str = ".") -> None:
    """Reject unsafe or existing output paths instead of deleting prior artifacts."""
    root_path = os.path.realpath(root)
    requested = os.path.abspath(
        path if os.path.isabs(path) else os.path.join(root_path, path)
    )
    resolved = os.path.realpath(requested)
    try:
        within_root = (
            os.path.commonpath((root_path, requested)) == root_path
            and os.path.commonpath((root_path, resolved)) == root_path
        )
    except ValueError:
        within_root = False
    protected = {
        root_path,
        os.path.realpath(os.path.expanduser("~")),
        _SOURCE_ROOT,
    }
    if (
        not within_root
        or resolved in protected
        or len(resolved) < 4
        or not os.path.basename(requested)
        or requested != resolved
    ):
        raise SystemExit(f"[!] Refusing unsafe output path {resolved} for {what}.")
    if os.path.lexists(requested) and os.path.islink(requested):
        raise SystemExit(
            f"[!] Refusing to write through symlinked output path {requested} "
            f"for {what}."
        )
    if os.path.lexists(requested):
        raise SystemExit(
            f"[!] Output path {requested} already exists; refusing to overwrite {what}. "
            "Choose a fresh HACKGRADER_RUN_ID or explicit output path."
        )


def verify_output_paths(
    run_dir: str,
    adapter_dir: str,
    merge_dir: str,
    *,
    root: str = ".",
    protected_paths: tuple[str, ...] = ("Modelfile",),
) -> None:
    """Check training outputs are fresh, safe, separate, and isolated from inputs."""
    paths = (
        (run_dir, "the trainer checkpoints"),
        (adapter_dir, "the adapter"),
        (merge_dir, "the merged checkpoint"),
    )
    root_path = os.path.realpath(root)

    def resolved_of(path: str) -> str:
        """Resolve one configured path against the project root."""
        absolute = path if os.path.isabs(path) else os.path.join(root_path, path)
        return os.path.normcase(os.path.realpath(absolute))

    resolved = [resolved_of(path) for path, _what in paths]
    protected = {
        resolved_of(path)
        for path in (*protected_paths, COMBINED_FILE, TRAIN_FILE, VAL_FILE, TEST_FILE)
    }
    for output in resolved:
        for input_path in protected:
            try:
                overlap = os.path.commonpath((output, input_path)) in {
                    output,
                    input_path,
                }
            except ValueError:
                overlap = False
            if overlap:
                raise SystemExit(
                    "[!] Trainer, adapter, and merge outputs may not overlap active "
                    f"datasets or protected files: {output} and {input_path}."
                )

    for path, what in paths:
        require_fresh_dir(path, what, root=root)
    for index, first in enumerate(resolved):
        for second in resolved[index + 1 :]:
            try:
                common = os.path.commonpath((first, second))
            except ValueError:
                common = ""
            if common in {first, second}:
                raise SystemExit(
                    "[!] Trainer, adapter, and merge outputs must use separate paths."
                )
