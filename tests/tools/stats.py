"""Timing and hardware summaries used by live integration tests.

Author: Quentin Bordelon
Date: 2026
License: MIT
"""

import platform
import time
from collections.abc import Callable

import GPUtil
import psutil


def benchmark_timing[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Decorate a test to print its elapsed runtime."""
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Run the wrapped test and print elapsed time around it."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        total_time = time.perf_counter() - start_time
        post_print()
        print(f"Total execution time: {total_time:.4f} seconds")
        post_print()
        return result

    return wrapper


def benchmark_stats[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Decorate a test to report the host OS, CPU, GPU, VRAM, and RAM."""
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Print machine information before running the wrapped test."""
        post_print()
        memory = psutil.virtual_memory()
        gpu = BlankGPU()
        stats = (
            f"OS: {platform.platform()}\nCPU: {platform.processor()}\n"
            f"GPU: {gpu.name}\nVRAM: {gpu.memoryTotal} MiB\n"
            f"RAM: {round(memory.total / (1024**3), 2)} GiB"
        )
        print(stats)
        post_print()
        return func(*args, **kwargs)

    return wrapper


def post_print() -> None:
    """Print a visual separator around test diagnostics."""
    print("\n----------------------------------------------------------------------\n")


class BlankGPU:
    """Represent the first available GPU, or provide a readable no-GPU fallback."""

    def __init__(self) -> None:
        """Read the first detected GPU's name and total VRAM."""
        self.name = "No GPU Detected"
        self.memoryTotal = 0
        gpus = GPUtil.getGPUs()
        if gpus:
            self.name = gpus[0].name
            self.memoryTotal = gpus[0].memoryTotal
