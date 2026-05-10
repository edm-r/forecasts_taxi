"""
src/optim/vectorize.py
----------------------
Calculs naïfs et vectorisés pour le pipeline NYC Taxi.

Partie 2 — Exercices 2.1 et 2.2
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_features_naive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Version volontairement non optimisée, en boucle Python.
    """

    result = df.copy()
    duration_minutes: list[float] = []
    avg_speed_mph: list[float] = []
    penalty: list[float] = []

    for row in result.itertuples(index=False):
        pickup = pd.Timestamp(row.tpep_pickup_datetime)
        dropoff = pd.Timestamp(row.tpep_dropoff_datetime)
        duration = (dropoff - pickup).total_seconds() / 60.0

        duration_minutes.append(duration)

        if duration > 0:
            avg_speed_mph.append(float(row.trip_distance) / (duration / 60.0))
        else:
            avg_speed_mph.append(0.0)

        fare_amount = float(row.fare_amount)
        fare_component = np.log1p(fare_amount) if fare_amount > -1.0 else np.nan
        penalty.append(0.5 * float(row.trip_distance) ** 2 + fare_component)

    result["duration_minutes"] = duration_minutes
    result["avg_speed_mph"] = avg_speed_mph
    result["penalty"] = penalty
    return result


def compute_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Réécriture vectorisée de ``compute_features_naive``.
    """

    result = df.copy()
    duration_minutes = (
        result["tpep_dropoff_datetime"] - result["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    trip_distance = result["trip_distance"].astype("float64")
    fare_amount = result["fare_amount"].astype("float64")
    hours = duration_minutes.to_numpy(dtype="float64") / 60.0
    distance_values = trip_distance.to_numpy(dtype="float64")
    fare_values = fare_amount.to_numpy(dtype="float64")

    avg_speed = np.divide(
        distance_values,
        hours,
        out=np.zeros_like(distance_values, dtype="float64"),
        where=hours > 0,
    )
    fare_component = np.full_like(fare_values, np.nan, dtype="float64")
    valid_fares = fare_values > -1.0
    fare_component[valid_fares] = np.log1p(fare_values[valid_fares])
    penalty = 0.5 * np.square(distance_values) + fare_component

    result["duration_minutes"] = duration_minutes.astype("float64")
    result["avg_speed_mph"] = avg_speed
    result["penalty"] = penalty
    return result


def build_centroid_distance_matrix(
    centroids: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
) -> np.ndarray:
    """
    Construit une matrice de distances euclidiennes sans boucle Python.
    """

    if x_col not in centroids.columns or y_col not in centroids.columns:
        raise KeyError(
            f"Les colonnes {x_col!r} et {y_col!r} doivent être présentes dans les centroïdes."
        )

    coords = centroids[[x_col, y_col]].to_numpy(dtype="float64")
    deltas = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(np.square(deltas), axis=2))


def recommend_optimized_dtypes(
    df: pd.DataFrame,
    *,
    category_ratio_threshold: float = 0.8,
) -> dict[str, str]:
    """
    Recommande des dtypes plus compacts sans modifier le DataFrame.
    """

    recommendations: dict[str, str] = {}

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_integer_dtype(series.dtype):
            recommendations[column] = _smallest_int_dtype(series)
            continue

        if pd.api.types.is_float_dtype(series.dtype):
            recommendations[column] = _recommended_float_dtype(series)
            continue

        if pd.api.types.is_object_dtype(series.dtype):
            non_null = series.dropna()
            if len(non_null) == 0:
                continue

            unique_ratio = non_null.nunique() / len(non_null)
            if unique_ratio <= category_ratio_threshold:
                recommendations[column] = "category"

    return recommendations


def apply_dtype_recommendations(
    df: pd.DataFrame,
    recommendations: dict[str, str],
) -> pd.DataFrame:
    """
    Applique un mapping de dtypes recommandé.
    """

    result = df.copy()
    for column, dtype in recommendations.items():
        result[column] = result[column].astype(dtype)
    return result


def memory_usage_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / (1024 ** 2))


def summarize_memory_optimization(
    df: pd.DataFrame,
    recommendations: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Produit un résumé mémoire avant/après recommandations.
    """

    recommendations = recommendations or recommend_optimized_dtypes(df)
    optimized = apply_dtype_recommendations(df, recommendations)

    before_mb = memory_usage_mb(df)
    after_mb = memory_usage_mb(optimized)
    gain_mb = before_mb - after_mb
    gain_pct = (gain_mb / before_mb * 100.0) if before_mb else 0.0

    return {
        "before_mb": before_mb,
        "after_mb": after_mb,
        "gain_mb": gain_mb,
        "gain_pct": gain_pct,
        "recommendations": recommendations,
    }


def _smallest_int_dtype(series: pd.Series) -> str:
    min_value = int(series.min())
    max_value = int(series.max())

    for dtype in (np.int8, np.int16, np.int32, np.int64):
        info = np.iinfo(dtype)
        if info.min <= min_value and max_value <= info.max:
            return np.dtype(dtype).name

    return "int64"


def _recommended_float_dtype(series: pd.Series) -> str:
    float32_series = series.astype("float32")
    if np.allclose(series.to_numpy(dtype="float64"), float32_series.to_numpy(dtype="float64"), equal_nan=True):
        return "float32"
    return "float64"
