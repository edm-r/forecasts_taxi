"""
src/core/context.py
-------------------
Gestionnaires de contexte pour le pipeline NYC Taxi.

Exercice 1.8 — Master 1 IABD
"""

from __future__ import annotations

import contextlib
import logging
import tracemalloc
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import pandas as pd


class Timer:
    """
    Context manager qui mesure et logue le temps d'un bloc.
    """

    def __init__(self, label: str, unit: str = "ms") -> None:
        self.label = label
        self.unit = unit
        self.elapsed = 0.0
        self._started_at = 0.0
        self._logger = logging.getLogger(__name__)

    def __enter__(self) -> "Timer":
        self._started_at = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed_s = perf_counter() - self._started_at
        factor = {"s": 1.0, "ms": 1000.0, "us": 1_000_000.0}.get(self.unit)
        if factor is None:
            raise ValueError(f"Unité non supportée: {self.unit!r}.")

        self.elapsed = elapsed_s * factor
        self._logger.info("[Timer] %s : %.3f %s", self.label, self.elapsed, self.unit)
        return False


class MemoryGuard:
    """
    Lève ``MemoryError`` si le pic mémoire du bloc dépasse un seuil.
    """

    def __init__(self, threshold_mb: float = 2000.0) -> None:
        self.threshold_mb = threshold_mb
        self.peak_mb = 0.0
        self._logger = logging.getLogger(__name__)

    def __enter__(self) -> "MemoryGuard":
        tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.peak_mb = peak_bytes / (1024 ** 2)
        self._logger.info(
            "[MemoryGuard] peak=%.3f MB threshold=%.3f MB",
            self.peak_mb,
            self.threshold_mb,
        )

        if exc_type is None and self.peak_mb > self.threshold_mb:
            raise MemoryError(
                f"Pic mémoire dépassé: {self.peak_mb:.3f} MB > {self.threshold_mb:.3f} MB."
            )
        return False


@contextlib.contextmanager
def temp_dtypes(df: pd.DataFrame, mapping: dict[str, Any]) -> Iterator[pd.DataFrame]:
    """
    Convertit temporairement des colonnes puis restaure leurs dtypes d'origine.
    """

    original_dtypes: dict[str, Any] = {}
    for column, dtype in mapping.items():
        if column not in df.columns:
            raise KeyError(f"Colonne absente: {column!r}.")
        original_dtypes[column] = df[column].dtype
        df[column] = df[column].astype(dtype)

    try:
        yield df
    finally:
        for column, dtype in original_dtypes.items():
            df[column] = df[column].astype(dtype)


@contextlib.contextmanager
def open_parquet_files(
    directory: str | Path,
    pattern: str = "*.parquet",
) -> Iterator[list[Any]]:
    """
    Ouvre tous les fichiers Parquet d'un répertoire en s'appuyant sur ``ExitStack``.
    """

    directory = Path(directory)
    paths = sorted(directory.glob(pattern))
    with ExitStack() as stack:
        handles = [stack.enter_context(path.open("rb")) for path in paths]
        yield handles
