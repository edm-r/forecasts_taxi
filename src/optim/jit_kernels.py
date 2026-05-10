"""
src/optim/jit_kernels.py
------------------------
Noyaux intensifs accélérés avec Numba.

Partie 2 — Exercice 2.6
"""

from __future__ import annotations

import math

import numba
import numpy as np


EARTH_RADIUS_KM = 6371.0088


def haversine_python(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Implémentation Python pure par boucle.
    """

    size = len(lat1)
    output = np.empty(size, dtype=np.float64)

    for index in range(size):
        output[index] = _haversine_scalar(
            float(lat1[index]),
            float(lon1[index]),
            float(lat2[index]),
            float(lon2[index]),
        )

    return output


def haversine_numpy(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Implémentation vectorisée NumPy.
    """

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


def haversine_numba(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    return _haversine_numba_core(lat1, lon1, lat2, lon2)


def haversine_numba_parallel(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    return _haversine_numba_parallel_core(lat1, lon1, lat2, lon2)


@numba.njit(cache=True)
def weighted_rolling_median_numba(
    values: np.ndarray,
    weights: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Médiane mobile pondérée compilée avec Numba.
    """

    if window <= 0:
        raise ValueError("window doit être > 0.")
    if len(weights) != window:
        raise ValueError("weights doit avoir la même taille que window.")

    output = np.empty(values.shape[0], dtype=np.float64)
    output[:] = np.nan

    for index in range(window - 1, values.shape[0]):
        window_values = values[index - window + 1 : index + 1].copy()
        window_weights = weights.copy()
        output[index] = _weighted_median_numba(window_values, window_weights)

    return output


def _haversine_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


@numba.njit(cache=True)
def _haversine_numba_core(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    output = np.empty(lat1.shape[0], dtype=np.float64)
    for index in range(lat1.shape[0]):
        output[index] = _haversine_numba_scalar(lat1[index], lon1[index], lat2[index], lon2[index])
    return output


@numba.njit(cache=True, parallel=True)
def _haversine_numba_parallel_core(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    output = np.empty(lat1.shape[0], dtype=np.float64)
    for index in numba.prange(lat1.shape[0]):
        output[index] = _haversine_numba_scalar(lat1[index], lon1[index], lat2[index], lon2[index])
    return output


@numba.njit(cache=True)
def _haversine_numba_scalar(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


@numba.njit(cache=True)
def _weighted_median_numba(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]

    total_weight = sorted_weights.sum()
    cumulative = 0.0
    half_weight = total_weight / 2.0

    for index in range(sorted_values.shape[0]):
        cumulative += sorted_weights[index]
        if cumulative >= half_weight:
            return sorted_values[index]

    return sorted_values[-1]
