"""Coordinate multi-GPU DDP training without changing the release sequence.

A DDP job runs one pipeline process per GPU. Every worker trains a full model
replica, so gradients stay in step; only the main worker scores validation, spends
the independent test, merges, and registers. The other workers follow the main
worker's decisions through small JSON files beside the run, which keeps a long
validation sweep from tripping a process-group timeout and keeps the release path
single-process exactly as it is today.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from typing import Any

GPU_ENV = "HACKGRADER_GPUS"
LOCAL_RANK_ENV = "LOCAL_RANK"
WORLD_SIZE_ENV = "WORLD_SIZE"
POLL_SECONDS = 5.0
_SINGLE = {"1", "single", "off", "false"}
_AUTOMATIC = {"", "auto", "all", "0"}


def requested_workers() -> int | None:
    """Return the configured worker count, or ``None`` when it is automatic."""
    raw = os.environ.get(GPU_ENV, "auto").strip().lower()
    if raw in _AUTOMATIC:
        return None
    if raw in _SINGLE:
        return 1
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(
            f"[!] {GPU_ENV} must be 'auto' or a GPU count; got {raw!r}. No training started."
        ) from None
    if value < 1:
        raise SystemExit(
            f"[!] {GPU_ENV} must be at least 1; got {value}. No training started."
        )
    return value


def active() -> bool:
    """Return whether this process is one worker of a multi-GPU job."""
    try:
        world = int(os.environ.get(WORLD_SIZE_ENV, "1"))
    except ValueError:
        return False
    return world > 1 and os.environ.get(LOCAL_RANK_ENV) is not None


def world_size() -> int:
    """Return the number of pipeline workers in this job."""
    try:
        return max(1, int(os.environ.get(WORLD_SIZE_ENV, "1")))
    except ValueError:
        return 1


def rank() -> int:
    """Return this worker's rank, which is zero outside a distributed job."""
    try:
        return max(0, int(os.environ.get("RANK", "0")))
    except ValueError:
        return 0


def local_rank() -> int:
    """Return the CUDA device index this worker trains on."""
    try:
        return max(0, int(os.environ.get(LOCAL_RANK_ENV, "0")))
    except ValueError:
        return 0


def is_main() -> bool:
    """Return whether this worker owns scoring, release, and the shared files."""
    return rank() == 0


def worker_directory() -> str:
    """Return the per-job directory the workers use for their own rendezvous."""
    run_id = os.environ.get("HACKGRADER_RUN_ID", "run")
    job = os.environ.get("TORCHELASTIC_RUN_ID", "job")
    return os.path.join(tempfile.gettempdir(), f"hackgrader-workers-{run_id}-{job}")


def rendezvous(*, what: str, timeout: float) -> None:
    """Announce one completed step on every worker before the main worker continues.

    Output paths are checked for freshness before the run directory exists, so every
    worker has to finish that check before the main worker creates it; otherwise a
    slower worker would find the directory already present and refuse the run.
    """
    if not active():
        return
    directory = worker_directory()
    os.makedirs(directory, exist_ok=True)
    with open(
        os.path.join(directory, f"ready-{rank()}.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({"rank": rank(), "pid": os.getpid()}, handle)
    if not is_main():
        return
    deadline = time.monotonic() + timeout
    while _ready_count(directory) < world_size():
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"the main worker waited {timeout / 60:.0f} minutes for {what}; "
                "not every worker reached it."
            )
        time.sleep(0.25)


def _ready_count(directory: str) -> int:
    """Count the workers that announced the current step."""
    try:
        return sum(
            1
            for entry in os.scandir(directory)
            if entry.name.startswith("ready-") and entry.name.endswith(".json")
        )
    except FileNotFoundError:
        return 0


def await_path(path: str, *, what: str, timeout: float) -> None:
    """Wait for the main worker to create a path, then continue without one."""
    if is_main():
        return
    deadline = time.monotonic() + timeout
    while not os.path.exists(path):
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"worker {rank()} waited {timeout / 3600:.1f}h for {what} at {path!r}; "
                "the main worker never wrote it."
            )
        time.sleep(POLL_SECONDS)


def prepare_round_directory(path: str, *, timeout: float) -> None:
    """Create one round's checkpoint directory on the main worker only."""
    if is_main():
        os.makedirs(path, exist_ok=False)
        return
    await_path(path, what="the round directory", timeout=timeout)


def shared(
    path: str, compute: Callable[[], dict[str, Any]], *, timeout: float
) -> dict[str, Any]:
    """Compute one decision on the main worker and give every worker the result."""
    if not active():
        return compute()
    if not is_main():
        return await_decision(path, timeout=timeout)
    try:
        payload = compute()
    except BaseException as error:
        publish_failure(path, error)
        raise
    publish(path, payload)
    return payload


def publish(path: str, payload: dict[str, Any]) -> None:
    """Write the main worker's decision where the other workers read it."""
    if active() and not is_main():
        raise RuntimeError("only the main worker publishes a decision")
    _write_json(path, payload)


def await_decision(path: str, *, timeout: float) -> dict[str, Any]:
    """Read the main worker's decision, failing when it reported a stop instead."""
    if not active():
        raise RuntimeError("no decision is published outside a multi-GPU job")
    return _await_decision(path, timeout)


def publish_failure(path: str, error: BaseException) -> None:
    """Record why the main worker stopped so the other workers stop too."""
    try:
        _write_json(
            path,
            {"failed": True, "error": f"{type(error).__name__}: {error}"},
        )
    except OSError:
        return


def _await_decision(path: str, timeout: float) -> dict[str, Any]:
    """Read the main worker's decision, stopping when it reports a failure."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            payload = None
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            if payload.get("failed"):
                raise RuntimeError(
                    f"the main worker stopped: {payload.get('error', 'unknown error')}"
                )
            return payload
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"worker {rank()} waited {timeout / 3600:.1f}h for {path!r}; "
                "the main worker never wrote it."
            )
        time.sleep(POLL_SECONDS)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    """Write one decision atomically so a reader never sees a partial file."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent or "."
    )
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def cuda_device_count() -> int:
    """Return the number of visible CUDA devices without initializing one when possible."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None and result.returncode == 0:
        listed = [line for line in result.stdout.splitlines() if line.strip()]
        if listed:
            return len(listed)
    try:
        import torch
    except (ImportError, OSError, RuntimeError):
        return 0
    try:
        return int(torch.cuda.device_count())
    except (RuntimeError, AttributeError):
        return 0


def nccl_available() -> bool:
    """Return whether this platform can run NCCL, which DDP with CUDA requires."""
    if os.name != "posix":
        return False
    try:
        import torch.distributed as dist
    except (ImportError, OSError, RuntimeError):
        return False
    try:
        return bool(dist.is_nccl_available())
    except (AttributeError, RuntimeError):
        return False


def worker_command(
    arguments: list[str], *, script: str | None = None
) -> list[str] | None:
    """Return the torchrun command that uses every GPU, or ``None`` to stay single-GPU."""
    if active():
        return None
    requested = requested_workers()
    if requested == 1:
        return None
    devices = cuda_device_count()
    if devices < 2:
        if requested is not None:
            print(
                f"[i] {GPU_ENV}={requested} but {devices} CUDA device(s) are visible; "
                "training uses one GPU."
            )
        return None
    if not nccl_available():
        print(
            "[i] Multi-GPU training needs NCCL, which this PyTorch build cannot use "
            f"here ({os.name}); training uses one GPU. Run the job on Linux for DDP."
        )
        return None
    workers = devices if requested is None else min(requested, devices)
    if workers < 2:
        return None
    # Every worker has to write to the same run; a per-process timestamp would split it.
    os.environ.setdefault("HACKGRADER_RUN_ID", time.strftime("%Y%m%d-%H%M%S"))
    print(
        f"[i] Training on {workers} GPUs: one full model replica per GPU, gradients "
        "synchronized with DDP; validation, testing, and release stay on worker 0."
    )
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node",
        str(workers),
        os.path.abspath(script or sys.argv[0]),
        *arguments,
    ]


def launch(command: list[str]) -> int:
    """Run the torchrun job in this process's place and return its exit status."""
    print("[i] " + " ".join(command), flush=True)
    try:
        completed = subprocess.run(command, check=False)
    except OSError as exc:
        raise SystemExit(
            f"[!] Multi-GPU launch failed ({type(exc).__name__}: {exc}); no training started."
        ) from None
    return completed.returncode
