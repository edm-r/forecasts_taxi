"""
src/features/temporal.py
------------------------
Feature engineering temporel pour le pipeline NYC Taxi.

Partie 3 — Exercice 3.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from core.metaclasses import BasePipelineStep

try:  # pragma: no cover - dépendance optionnelle
    from statsmodels.tsa.seasonal import STL
except ImportError:  # pragma: no cover
    STL = None


class TemporalFeaturizer(BasePipelineStep):
    """
    Ajoute des variables calendaires, cycliques et des indicateurs proxy.
    """

    name = "temporal_featurizer"

    def __init__(
        self,
        *,
        datetime_col: str = "tpep_pickup_datetime",
        prefix: str = "pickup",
        add_cyclical: bool = True,
    ) -> None:
        self.datetime_col = datetime_col
        self.prefix = prefix
        self.add_cyclical = add_cyclical

    def _run(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.datetime_col not in df.columns:
            raise KeyError(f"Colonne datetime absente: {self.datetime_col!r}.")

        result = df.copy()
        dt = pd.to_datetime(result[self.datetime_col], errors="coerce")
        if dt.isna().any():
            raise ValueError(
                f"La colonne {self.datetime_col!r} contient des dates invalides."
            )

        prefix = self.prefix
        result[f"{prefix}_hour"] = dt.dt.hour.astype("int8")
        result[f"{prefix}_weekday"] = dt.dt.weekday.astype("int8")
        result[f"{prefix}_day"] = dt.dt.day.astype("int8")
        result[f"{prefix}_month"] = dt.dt.month.astype("int8")
        result[f"{prefix}_quarter"] = dt.dt.quarter.astype("int8")
        result[f"{prefix}_iso_week"] = dt.dt.isocalendar().week.astype("int16")
        result["is_weekend"] = (dt.dt.weekday >= 5)
        result["is_us_holiday"] = _compute_us_holiday_flag(dt)

        weekday = result[f"{prefix}_weekday"]
        hour = result[f"{prefix}_hour"]
        result["is_rush_hour"] = (
            weekday.lt(5)
            & (hour.between(7, 9) | hour.between(17, 19))
        )
        result["is_late_night"] = hour.between(0, 5)

        if self.add_cyclical:
            result = add_cyclical_features(
                result,
                columns={
                    f"{prefix}_hour": 24,
                    f"{prefix}_weekday": 7,
                    f"{prefix}_month": 12,
                },
            )

        return result


def add_cyclical_features(
    df: pd.DataFrame,
    *,
    columns: dict[str, int],
) -> pd.DataFrame:
    """
    Ajoute sin/cos pour les colonnes circulaires indiquées.
    """

    result = df.copy()
    for column, period in columns.items():
        if column not in result.columns:
            raise KeyError(f"Colonne absente pour encodage cyclique: {column!r}.")

        angle = 2 * np.pi * result[column].astype("float64") / float(period)
        result[f"{column}_sin"] = np.sin(angle)
        result[f"{column}_cos"] = np.cos(angle)

    return result


def aggregate_hourly_trip_counts(
    df: pd.DataFrame,
    *,
    datetime_col: str = "tpep_pickup_datetime",
) -> pd.Series:
    """
    Agrège le dataset à la maille horaire sur le nombre de courses.
    """

    if datetime_col not in df.columns:
        raise KeyError(f"Colonne datetime absente: {datetime_col!r}.")

    dt = pd.to_datetime(df[datetime_col], errors="coerce")
    if dt.isna().any():
        raise ValueError(
            f"La colonne {datetime_col!r} contient des dates invalides."
        )

    hourly = (
        pd.DataFrame({"pickup_hour": dt.dt.floor("h")})
        .groupby("pickup_hour")
        .size()
        .rename("trip_count")
        .sort_index()
    )
    return hourly


def decompose_hourly_seasonality(
    df: pd.DataFrame,
    *,
    datetime_col: str = "tpep_pickup_datetime",
    periods: dict[str, int] | None = None,
    robust: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Décompose la série horaire avec STL pour plusieurs périodicités.
    """

    if STL is None:
        raise RuntimeError(
            "statsmodels est requis pour la décomposition STL. "
            "Ajoutez 'statsmodels' aux dépendances du projet."
        )

    periods = periods or {"daily": 24, "weekly": 24 * 7}
    hourly = aggregate_hourly_trip_counts(df, datetime_col=datetime_col)

    results: dict[str, pd.DataFrame] = {}
    for name, period in periods.items():
        if len(hourly) < period * 2:
            raise ValueError(
                f"La série horaire est trop courte pour une STL de période {period}."
            )

        stl = STL(hourly.astype("float64"), period=period, robust=robust)
        fitted = stl.fit()
        results[name] = pd.DataFrame(
            {
                "observed": hourly,
                "trend": fitted.trend,
                "seasonal": fitted.seasonal,
                "resid": fitted.resid,
            }
        )

    return results


def _compute_us_holiday_flag(dt: pd.Series) -> pd.Series:
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=dt.min().normalize(), end=dt.max().normalize())
    holiday_days = holidays.normalize()
    return dt.dt.normalize().isin(holiday_days)
