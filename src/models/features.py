"""
src/models/features.py
----------------------
Préparation des features communes à l'entraînement et au serving.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


RAW_TAXI_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "tpep_pickup_datetime",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "tip_amount",
]

CANONICAL_REQUEST_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "pickup_datetime",
    "pu_location_id",
    "do_location_id",
    "payment_type",
]

MODEL_FEATURE_COLUMNS = [
    "passenger_count",
    "trip_distance",
    "pu_location_id",
    "do_location_id",
    "payment_type",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
    "is_weekend",
    "is_rush_hour",
    "is_late_night",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "pickup_weekday_sin",
    "pickup_weekday_cos",
    "pickup_month_sin",
    "pickup_month_cos",
]


def canonicalize_raw_taxi_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise les colonnes brutes TLC vers le schéma de serving.
    """

    missing = set(RAW_TAXI_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"Colonnes brutes manquantes: {sorted(missing)}")

    result = df[RAW_TAXI_COLUMNS].rename(
        columns={
            "tpep_pickup_datetime": "pickup_datetime",
            "PULocationID": "pu_location_id",
            "DOLocationID": "do_location_id",
        }
    )
    return result


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la matrice de features numériques utilisée par le modèle.
    """

    missing = set(CANONICAL_REQUEST_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"Colonnes d'entrée manquantes: {sorted(missing)}")

    result = df[CANONICAL_REQUEST_COLUMNS].copy()
    dt = pd.to_datetime(result["pickup_datetime"], errors="coerce")
    if dt.isna().any():
        raise ValueError("pickup_datetime contient des dates invalides.")

    hour = dt.dt.hour.astype("int16")
    weekday = dt.dt.weekday.astype("int16")
    month = dt.dt.month.astype("int16")

    result["pickup_hour"] = hour
    result["pickup_weekday"] = weekday
    result["pickup_month"] = month
    result["is_weekend"] = (weekday >= 5).astype("int8")
    result["is_rush_hour"] = (
        (weekday < 5)
        & ((hour.between(7, 9)) | (hour.between(17, 19)))
    ).astype("int8")
    result["is_late_night"] = hour.between(0, 5).astype("int8")

    result["pickup_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    result["pickup_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    result["pickup_weekday_sin"] = np.sin(2.0 * np.pi * weekday / 7.0)
    result["pickup_weekday_cos"] = np.cos(2.0 * np.pi * weekday / 7.0)
    result["pickup_month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    result["pickup_month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    numeric = result[MODEL_FEATURE_COLUMNS].copy()
    return numeric.astype("float64")


def prepare_training_frame(
    raw_df: pd.DataFrame,
    *,
    tip_pct_clip: tuple[float, float] = (0.0, 2.0),
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Construit X/y à partir des colonnes brutes TLC.

    Retourne :
    - X : features numériques prêtes pour le modèle
    - y : cible tip_pct
    - cleaned : DataFrame canonique nettoyé, utile pour l'audit
    """

    canonical = canonicalize_raw_taxi_frame(raw_df)
    cleaned = canonical.copy()

    dt = pd.to_datetime(cleaned["pickup_datetime"], errors="coerce")
    tip_pct = cleaned["tip_amount"] / cleaned["fare_amount"]

    valid_mask = (
        dt.notna()
        & cleaned["fare_amount"].gt(0)
        & cleaned["tip_amount"].ge(0)
        & cleaned["trip_distance"].between(0.1, 100.0)
        & cleaned["passenger_count"].between(1, 8)
        & cleaned["payment_type"].isin([1, 2, 3, 4])
        & cleaned["pu_location_id"].between(1, 265)
        & cleaned["do_location_id"].between(1, 265)
        & tip_pct.between(*tip_pct_clip)
    )

    cleaned = cleaned.loc[valid_mask].copy()
    cleaned["pickup_datetime"] = pd.to_datetime(cleaned["pickup_datetime"], errors="raise")
    cleaned["tip_pct"] = (cleaned["tip_amount"] / cleaned["fare_amount"]).astype("float64")

    X = build_feature_matrix(cleaned)
    y = cleaned["tip_pct"].astype("float64")
    return X, y, cleaned


def frame_from_requests(records: Iterable[dict]) -> pd.DataFrame:
    """
    Construit un DataFrame canonique à partir des payloads API.
    """

    return pd.DataFrame(list(records), columns=CANONICAL_REQUEST_COLUMNS)
