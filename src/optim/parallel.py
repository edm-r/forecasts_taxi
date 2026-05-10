"""
src/optim/parallel.py
---------------------
Parallélisme CPU et I/O pour le pipeline NYC Taxi.

Partie 2 — Exercices 2.3 et 2.4
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import pandas as pd

from optim.vectorize import compute_features_vectorized


def process_months_sequential(
    months: Iterable[int | tuple[int, int]],
    *,
    year: int = 2023,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
) -> list[Path]:
    """
    Version séquentielle du traitement par mois.
    """

    return [
        _process_single_month(task)
        for task in _normalize_month_tasks(months, year=year, raw_dir=raw_dir, processed_dir=processed_dir)
    ]


def process_months_parallel(
    months: Iterable[int | tuple[int, int]],
    *,
    year: int = 2023,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    max_workers: int | None = None,
) -> list[Path]:
    """
    Traite plusieurs mois en parallèle via ``ProcessPoolExecutor``.
    """

    tasks = _normalize_month_tasks(months, year=year, raw_dir=raw_dir, processed_dir=processed_dir)
    if not tasks:
        return []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_process_single_month, tasks))


def benchmark_month_processing(
    months: Iterable[int | tuple[int, int]],
    *,
    year: int = 2023,
    raw_dir: str | Path = "data/raw",
    processed_dir: str | Path = "data/processed",
    max_workers: int | None = None,
) -> dict[str, float]:
    """
    Compare le temps séquentiel et parallèle sur la même liste de mois.
    """

    started_at = perf_counter()
    process_months_sequential(months, year=year, raw_dir=raw_dir, processed_dir=processed_dir)
    sequential_s = perf_counter() - started_at

    started_at = perf_counter()
    process_months_parallel(
        months,
        year=year,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        max_workers=max_workers,
    )
    parallel_s = perf_counter() - started_at

    speedup = (sequential_s / parallel_s) if parallel_s else math.inf
    return {
        "sequential_s": sequential_s,
        "parallel_s": parallel_s,
        "speedup": speedup,
    }


def run_cpu_bound_with_threads(
    *,
    task_count: int = 4,
    work_size: int = 200_000,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """
    Exécute un calcul CPU-bound dans un ``ThreadPoolExecutor``.
    """

    return _run_cpu_bound(ThreadPoolExecutor, task_count=task_count, work_size=work_size, max_workers=max_workers)


def run_cpu_bound_with_processes(
    *,
    task_count: int = 4,
    work_size: int = 200_000,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """
    Exécute un calcul CPU-bound dans un ``ProcessPoolExecutor``.
    """

    return _run_cpu_bound(ProcessPoolExecutor, task_count=task_count, work_size=work_size, max_workers=max_workers)


def read_parquet_files_sequential(
    paths: Iterable[str | Path],
    *,
    columns: list[str] | None = None,
) -> list[pd.DataFrame]:
    return [pd.read_parquet(path, columns=columns) for path in paths]


def read_parquet_files_threaded(
    paths: Iterable[str | Path],
    *,
    columns: list[str] | None = None,
    max_workers: int | None = None,
) -> list[pd.DataFrame]:
    path_list = [Path(path) for path in paths]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_read_single_parquet, path_list, [columns] * len(path_list)))


def benchmark_io_readers(
    paths: Iterable[str | Path],
    *,
    columns: list[str] | None = None,
    max_workers: int | None = None,
) -> dict[str, float]:
    path_list = list(paths)

    started_at = perf_counter()
    read_parquet_files_sequential(path_list, columns=columns)
    sequential_s = perf_counter() - started_at

    started_at = perf_counter()
    read_parquet_files_threaded(path_list, columns=columns, max_workers=max_workers)
    threaded_s = perf_counter() - started_at

    return {
        "sequential_s": sequential_s,
        "threaded_s": threaded_s,
        "speedup": (sequential_s / threaded_s) if threaded_s else math.inf,
    }


def _normalize_month_tasks(
    months: Iterable[int | tuple[int, int]],
    *,
    year: int,
    raw_dir: str | Path,
    processed_dir: str | Path,
) -> list[dict[str, Any]]:
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    tasks: list[dict[str, Any]] = []

    for item in months:
        month_year, month = item if isinstance(item, tuple) else (year, item)
        tasks.append(
            {
                "year": int(month_year),
                "month": int(month),
                "raw_dir": str(raw_dir),
                "processed_dir": str(processed_dir),
            }
        )

    return tasks


def _process_single_month(task: dict[str, Any]) -> Path:
    year = task["year"]
    month = task["month"]
    raw_dir = Path(task["raw_dir"])
    processed_dir = Path(task["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_path = raw_dir / f"yellow_tripdata_{year}-{month:02d}.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier mensuel introuvable: {input_path}")

    df = pd.read_parquet(input_path)
    featured = compute_features_vectorized(df)

    output_path = processed_dir / f"yellow_tripdata_{year}-{month:02d}_features.parquet"
    featured.to_parquet(output_path, index=False)
    return output_path


def _cpu_bound_chunk(task: tuple[int, int]) -> float:
    start, stop = task
    total = 0.0
    for value in range(start, stop):
        total += math.sin(value) ** 2 + math.cos(value) ** 2
    return total


def _run_cpu_bound(
    executor_cls,
    *,
    task_count: int,
    work_size: int,
    max_workers: int | None,
) -> dict[str, Any]:
    tasks = [
        (index * work_size, (index + 1) * work_size)
        for index in range(task_count)
    ]

    started_at = perf_counter()
    with executor_cls(max_workers=max_workers) as executor:
        results = list(executor.map(_cpu_bound_chunk, tasks))
    elapsed_s = perf_counter() - started_at

    return {
        "results": results,
        "total": sum(results),
        "elapsed_s": elapsed_s,
    }


def _read_single_parquet(path: Path, columns: list[str] | None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)
