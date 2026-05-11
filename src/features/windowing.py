"""
src/features/windowing.py
------------------------
Window functions et time-series engineering pour NYC Taxi.

Partie 3 — Exercices 3.3 et 3.8
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - dépendance optionnelle
    from statsmodels.tsa.stattools import adfuller
except ImportError:  # pragma: no cover
    adfuller = None


def add_zone_previous_tip_mean(
    df: pd.DataFrame,
    *,
    group_col: str = "PULocationID",
    time_col: str = "tpep_pickup_datetime",
    tip_col: str = "tip_amount",
    window: int = 5,
    output_col: str = "tip_amount_prev_5_mean",
) -> pd.DataFrame:
    result, sorted_df = _prepare_temporal_grouped(df, group_col=group_col, time_col=time_col)
    feature = (
        sorted_df.groupby(group_col, observed=False)[tip_col]
        .transform(lambda series: series.shift(1).rolling(window, min_periods=1).mean())
    )
    result[output_col] = feature.reindex(result.index)
    return result


def add_fare_rank_pct(
    df: pd.DataFrame,
    *,
    group_col: str = "PULocationID",
    time_col: str = "tpep_pickup_datetime",
    fare_col: str = "fare_amount",
    output_col: str = "fare_amount_rank_pct",
) -> pd.DataFrame:
    result = df.copy()
    dt = pd.to_datetime(result[time_col], errors="coerce")
    if dt.isna().any():
        raise ValueError(f"Dates invalides dans {time_col!r}.")

    day_key = dt.dt.normalize()
    result[output_col] = (
        result.groupby([group_col, day_key], observed=False)[fare_col]
        .rank(pct=True)
        .astype("float64")
    )
    return result


def add_expanding_tip_pct_mean(
    df: pd.DataFrame,
    *,
    group_col: str = "PULocationID",
    time_col: str = "tpep_pickup_datetime",
    tip_pct_col: str = "tip_pct",
    output_col: str = "tip_pct_expanding_mean",
) -> pd.DataFrame:
    result, sorted_df = _prepare_temporal_grouped(df, group_col=group_col, time_col=time_col)
    feature = (
        sorted_df.groupby(group_col, observed=False)[tip_pct_col]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )
    result[output_col] = feature.reindex(result.index)
    return result


def compute_zone_daily_trip_lags(
    df: pd.DataFrame,
    *,
    group_col: str = "PULocationID",
    time_col: str = "tpep_pickup_datetime",
    lags: Iterable[int] = (1, 3, 7),
) -> pd.DataFrame:
    dt = pd.to_datetime(df[time_col], errors="coerce")
    if dt.isna().any():
        raise ValueError(f"Dates invalides dans {time_col!r}.")

    daily = (
        df.assign(service_day=dt.dt.normalize())
        .groupby([group_col, "service_day"], observed=False)
        .size()
        .rename("trip_count")
        .reset_index()
    )

    completed_groups: list[pd.DataFrame] = []
    for zone, zone_df in daily.groupby(group_col, observed=False):
        zone_df = zone_df.sort_values("service_day").set_index("service_day")
        full_range = pd.date_range(zone_df.index.min(), zone_df.index.max(), freq="D")
        zone_reindexed = zone_df.reindex(full_range)
        zone_reindexed[group_col] = zone
        zone_reindexed["trip_count"] = zone_reindexed["trip_count"].fillna(0)
        zone_reindexed.index.name = "service_day"
        completed_groups.append(zone_reindexed.reset_index())

    result = pd.concat(completed_groups, ignore_index=True)
    result = result.rename(columns={"index": "service_day"})
    result = result.sort_values([group_col, "service_day"]).reset_index(drop=True)

    for lag in lags:
        result[f"trip_count_lag_{lag}d"] = (
            result.groupby(group_col, observed=False)["trip_count"].shift(lag)
        )

    return result


def trimmed_mean(series: pd.Series, proportion: float = 0.1) -> float:
    if not 0.0 <= proportion < 0.5:
        raise ValueError("proportion doit être dans [0, 0.5).")

    clean = series.dropna().sort_values().to_numpy(dtype="float64")
    if clean.size == 0:
        return float("nan")

    trim = int(clean.size * proportion)
    if trim == 0:
        return float(clean.mean())

    trimmed = clean[trim:-trim]
    if trimmed.size == 0:
        return float(clean.mean())
    return float(trimmed.mean())


def aggregate_hourly_zone_timeseries(
    df: pd.DataFrame,
    *,
    zone_col: str = "PULocationID",
    time_col: str = "tpep_pickup_datetime",
    tip_col: str = "tip_amount",
    fare_col: str = "fare_amount",
) -> pd.DataFrame:
    dt = pd.to_datetime(df[time_col], errors="coerce")
    if dt.isna().any():
        raise ValueError(f"Dates invalides dans {time_col!r}.")

    result = (
        df.assign(pickup_hour=dt.dt.floor("h"))
        .groupby([zone_col, "pickup_hour"], observed=False)
        .agg(
            trip_count=(zone_col, "size"),
            tip_mean=(tip_col, "mean"),
            fare_mean=(fare_col, "mean"),
        )
        .reset_index()
        .sort_values([zone_col, "pickup_hour"])
        .reset_index(drop=True)
    )
    return result


def add_timeseries_lag_features(
    df: pd.DataFrame,
    *,
    group_col: str = "PULocationID",
    time_col: str = "pickup_hour",
    value_cols: Iterable[str] = ("trip_count", "tip_mean", "fare_mean"),
    lags: Iterable[int] = (1, 24, 168),
) -> pd.DataFrame:
    result = _sort_group_time(df, group_col=group_col, time_col=time_col)
    for value_col in value_cols:
        for lag in lags:
            result[f"{value_col}_lag_{lag}"] = (
                result.groupby(group_col, observed=False)[value_col].shift(lag)
            )
    return result


def add_timeseries_rolling_features(
    df: pd.DataFrame,
    *,
    group_col: str = "PULocationID",
    time_col: str = "pickup_hour",
    value_cols: Iterable[str] = ("trip_count",),
    windows: Iterable[int] = (3, 24, 168),
) -> pd.DataFrame:
    result = _sort_group_time(df, group_col=group_col, time_col=time_col)
    for value_col in value_cols:
        shifted = result.groupby(group_col, observed=False)[value_col].shift(1)
        for window in windows:
            grouped = shifted.groupby(result[group_col], sort=False)
            result[f"{value_col}_rolling_mean_{window}"] = grouped.transform(
                lambda series: series.rolling(window, min_periods=1).mean()
            )
            result[f"{value_col}_rolling_std_{window}"] = grouped.transform(
                lambda series: series.rolling(window, min_periods=1).std()
            )
            result[f"{value_col}_rolling_min_{window}"] = grouped.transform(
                lambda series: series.rolling(window, min_periods=1).min()
            )
            result[f"{value_col}_rolling_max_{window}"] = grouped.transform(
                lambda series: series.rolling(window, min_periods=1).max()
            )
    return result


def add_timeseries_differences(
    df: pd.DataFrame,
    *,
    group_col: str = "PULocationID",
    time_col: str = "pickup_hour",
    value_cols: Iterable[str] = ("trip_count",),
    periods: Iterable[int] = (1, 24),
) -> pd.DataFrame:
    result = _sort_group_time(df, group_col=group_col, time_col=time_col)
    for value_col in value_cols:
        for period in periods:
            result[f"{value_col}_diff_{period}"] = (
                result.groupby(group_col, observed=False)[value_col].diff(period)
            )
    return result


def add_fourier_power_features(
    df: pd.DataFrame,
    *,
    group_col: str = "PULocationID",
    time_col: str = "pickup_hour",
    value_col: str = "trip_count",
    history_hours: int = 24 * 24,
    top_k: int = 5,
) -> pd.DataFrame:
    result = _sort_group_time(df, group_col=group_col, time_col=time_col)
    for harmonic in range(1, top_k + 1):
        result[f"{value_col}_fourier_power_{harmonic}"] = np.nan

    for zone, zone_df in result.groupby(group_col, observed=False):
        values = zone_df[value_col].to_numpy(dtype="float64")
        zone_index = zone_df.index.to_numpy()

        for idx in range(history_hours - 1, len(zone_df)):
            window = values[idx - history_hours + 1 : idx + 1]
            fft = np.fft.rfft(window)
            power = np.square(np.abs(fft))[1 : top_k + 1]
            for harmonic, value in enumerate(power, start=1):
                result.loc[zone_index[idx], f"{value_col}_fourier_power_{harmonic}"] = value

    return result


def adf_stationarity_report(series: pd.Series) -> dict[str, float]:
    if adfuller is None:
        raise RuntimeError(
            "statsmodels est requis pour le test ADF. Ajoutez 'statsmodels' aux dépendances."
        )

    cleaned = series.dropna().astype("float64")
    statistic, pvalue, usedlag, nobs, critical_values, icbest = adfuller(cleaned)
    return {
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "usedlag": float(usedlag),
        "nobs": float(nobs),
        "critical_1pct": float(critical_values["1%"]),
        "critical_5pct": float(critical_values["5%"]),
        "critical_10pct": float(critical_values["10%"]),
        "icbest": float(icbest),
    }


def _prepare_temporal_grouped(
    df: pd.DataFrame,
    *,
    group_col: str,
    time_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()
    result[time_col] = pd.to_datetime(result[time_col], errors="coerce")
    if result[time_col].isna().any():
        raise ValueError(f"Dates invalides dans {time_col!r}.")
    sorted_df = result.sort_values([group_col, time_col]).copy()
    return result, sorted_df


def _sort_group_time(
    df: pd.DataFrame,
    *,
    group_col: str,
    time_col: str,
) -> pd.DataFrame:
    result = df.copy()
    result[time_col] = pd.to_datetime(result[time_col], errors="coerce")
    if result[time_col].isna().any():
        raise ValueError(f"Dates invalides dans {time_col!r}.")
    return result.sort_values([group_col, time_col]).reset_index(drop=True)
