"""Protect existing checkpoints and active inputs from resume outputs."""

import os

from . import preflight


def verify_resume_outputs(
    run_dir: str,
    adapter_dir: str,
    merge_dir: str,
    active_paths: tuple[str, ...],
    *,
    root: str = ".",
) -> None:
    """Keep release outputs fresh and separate from the old run and active inputs."""
    root_path = os.path.realpath(root)
    run = os.path.realpath(
        run_dir if os.path.isabs(run_dir) else os.path.join(root_path, run_dir)
    )
    try:
        if os.path.commonpath((root_path, run)) != root_path:
            raise SystemExit("[!] Resume run must stay inside the model workspace.")
    except ValueError:
        raise SystemExit(
            "[!] Resume run must stay inside the model workspace."
        ) from None
    protected = [
        run,
        *(
            os.path.realpath(
                path if os.path.isabs(path) else os.path.join(root_path, path)
            )
            for path in active_paths
        ),
    ]
    outputs = (adapter_dir, merge_dir)
    resolved = [
        os.path.realpath(path if os.path.isabs(path) else os.path.join(root_path, path))
        for path in outputs
    ]
    for output, destination in zip(outputs, resolved, strict=True):
        for source in protected:
            try:
                overlap = os.path.commonpath((destination, source)) in {
                    destination,
                    source,
                }
            except ValueError:
                overlap = False
            if overlap:
                raise SystemExit(
                    "[!] Resume outputs may not overlap old runs or active data."
                )
        preflight.require_fresh_dir(output, "resume output", root=root_path)
    try:
        common = os.path.commonpath(resolved)
    except ValueError:
        common = ""
    if common in set(resolved):
        raise SystemExit("[!] Resume adapter and merge outputs must be separate.")
