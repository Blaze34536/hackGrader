import platform
import time
from collections.abc import Callable

import GPUtil
import psutil


def benchmark_timing[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time: float = time.perf_counter()
        result: R = func(*args, **kwargs)
        end_time: float = time.perf_counter()
        total_time: float = end_time - start_time

        post_print()
        print(f"Total execution time: {total_time:.4f} seconds")
        post_print()
        return result

    return wrapper


def benchmark_stats[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        post_print()
        memory = psutil.virtual_memory()
        gpu = BlankGPU()

        stats = f"OS: {platform.platform()}\nCPU: {platform.processor()}\nGPU: {gpu.name}\nVRAM: {gpu.memoryTotal} MiB\nRAM: {round(memory.total / (1024**3), 2)} GiB"
        print(stats)
        post_print()

        return func(*args, **kwargs)

    return wrapper


def post_print() -> None:
    print("\n----------------------------------------------------------------------\n")


class BlankGPU:
    def __init__(self) -> None:
        self.name: str = "No GPU Detected"
        self.memoryTotal: int = 0

        gpus = GPUtil.getGPUs()
        if gpus:
            self.name = gpus[0].name
            self.memoryTotal = gpus[0].memoryTotal
