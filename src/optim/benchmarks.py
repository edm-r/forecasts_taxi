"""
src/optim/benchmarks.py
-----------------------
Helpers de benchmark et profilage léger pour la partie 2.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import tracemalloc
from statistics import mean
from time import perf_counter
from typing import Any, Callable


def time_call(
    func: Callable[..., Any],
    *args: Any,
    repeats: int = 3,
    warmup: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Mesure le temps d'exécution moyen d'une fonction.
    """

    if repeats < 1:
        raise ValueError("repeats doit être >= 1.")
    if warmup < 0:
        raise ValueError("warmup doit être >= 0.")

    for _ in range(warmup):
        func(*args, **kwargs)

    samples_ms: list[float] = []
    result: Any = None
    for _ in range(repeats):
        started_at = perf_counter()
        result = func(*args, **kwargs)
        samples_ms.append((perf_counter() - started_at) * 1000.0)

    return {
        "function": func.__name__,
        "samples_ms": samples_ms,
        "mean_ms": mean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "result": result,
    }


def profile_with_cprofile(
    func: Callable[..., Any],
    *args: Any,
    sort_by: str = "cumulative",
    top_n: int = 10,
    **kwargs: Any,
) -> str:
    """
    Retourne le rapport texte cProfile d'une fonction.
    """

    profiler = cProfile.Profile()
    profiler.enable()
    func(*args, **kwargs)
    profiler.disable()

    buffer = io.StringIO()
    stats = pstats.Stats(profiler, stream=buffer).sort_stats(sort_by)
    stats.print_stats(top_n)
    return buffer.getvalue()


def measure_peak_memory(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Mesure le pic mémoire Python observé via tracemalloc.
    """

    tracemalloc.start()
    result = func(*args, **kwargs)
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "function": func.__name__,
        "current_mb": current_bytes / (1024 ** 2),
        "peak_mb": peak_bytes / (1024 ** 2),
        "result": result,
    }


def compare_functions(
    functions: list[Callable[..., Any]],
    *args: Any,
    repeats: int = 3,
    warmup: int = 1,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Compare plusieurs fonctions sur la même entrée.
    """

    return [
        time_call(func, *args, repeats=repeats, warmup=warmup, **kwargs)
        for func in functions
    ]
