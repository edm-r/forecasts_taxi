"""
src/features/encoding.py
------------------------
Encodage avancé, optimisation mémoire et tableaux d'activité.

Partie 3 — Exercices 3.1, 3.2, 3.4 et 3.6
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import KFold


def optimize_dtypes(
    df: pd.DataFrame,
    *,
    category_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Réduit l'empreinte mémoire via downcasting et catégories.
    """

    result = df.copy()

    for column in result.columns:
        series = result[column]

        if pd.api.types.is_integer_dtype(series.dtype):
            result[column] = pd.to_numeric(series, downcast="integer")
            continue

        if pd.api.types.is_float_dtype(series.dtype):
            downcasted = pd.to_numeric(series, downcast="float")
            if np.allclose(
                series.to_numpy(dtype="float64"),
                downcasted.to_numpy(dtype="float64"),
                equal_nan=True,
            ):
                result[column] = downcasted
            continue

        if pd.api.types.is_object_dtype(series.dtype):
            non_null = series.dropna()
            if len(non_null) == 0:
                continue

            unique_ratio = non_null.nunique() / len(non_null)
            # Les faibles cardinalités profitent fortement du dtype category,
            # même lorsque l'échantillon est petit et que le ratio reste élevé.
            if unique_ratio <= category_threshold or non_null.nunique() <= 50:
                result[column] = series.astype("category")

    return result


def summarize_dtype_optimization(df: pd.DataFrame) -> dict[str, Any]:
    optimized = optimize_dtypes(df)
    before_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    after_mb = optimized.memory_usage(deep=True).sum() / (1024 ** 2)
    return {
        "before_mb": float(before_mb),
        "after_mb": float(after_mb),
        "gain_mb": float(before_mb - after_mb),
        "gain_pct": float(((before_mb - after_mb) / before_mb * 100.0) if before_mb else 0.0),
        "optimized": optimized,
    }


def iter_parquet_batches(
    path: str | Path,
    *,
    batch_size: int = 100_000,
    columns: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """
    Lit un Parquet par batches avec PyArrow.
    """

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def compute_streaming_statistics(
    path: str | Path,
    *,
    batch_size: int = 100_000,
    fare_col: str = "fare_amount",
    tip_col: str = "tip_amount",
) -> dict[str, float]:
    """
    Calcule quelques statistiques sans charger tout le fichier en mémoire.
    """
    parquet_file = pq.ParquetFile(path)
    available_columns = set(parquet_file.schema.names)
    requested_columns = {fare_col, tip_col}
    missing_columns = requested_columns - available_columns
    if missing_columns:
        raise KeyError(f"Colonnes absentes du parquet: {sorted(missing_columns)}")

    row_count = 0
    fare_sum = 0.0
    tip_sum = 0.0

    for batch_df in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=[fare_col, tip_col],
    ):
        batch_df = batch_df.to_pandas()
        row_count += len(batch_df)
        fare_sum += float(batch_df[fare_col].sum())
        tip_sum += float(batch_df[tip_col].sum())

    return {
        "row_count": float(row_count),
        "fare_sum": fare_sum,
        "tip_sum": tip_sum,
        "tip_mean": (tip_sum / row_count) if row_count else 0.0,
    }


def build_activity_dashboard(
    df: pd.DataFrame,
    *,
    pickup_col: str = "PULocationID",
    datetime_col: str = "tpep_pickup_datetime",
    revenue_col: str = "total_amount",
    tip_pct_col: str = "tip_pct",
) -> pd.DataFrame:
    """
    Produit la table d'activité MultiIndex (zone, heure, weekday).
    """

    if not {pickup_col, datetime_col, revenue_col, tip_pct_col}.issubset(df.columns):
        missing = {pickup_col, datetime_col, revenue_col, tip_pct_col} - set(df.columns)
        raise KeyError(f"Colonnes manquantes: {sorted(missing)}")

    dt = pd.to_datetime(df[datetime_col], errors="coerce")
    if dt.isna().any():
        raise ValueError(f"La colonne {datetime_col!r} contient des dates invalides.")

    enriched = df.copy()
    enriched["hour"] = dt.dt.hour.astype("int8")
    enriched["weekday"] = dt.dt.weekday.astype("int8")

    activity = (
        enriched.groupby([pickup_col, "hour", "weekday"], observed=False)
        .agg(
            trip_count=(pickup_col, "size"),
            revenue_total=(revenue_col, "sum"),
            tip_pct_mean=(tip_pct_col, "mean"),
        )
        .sort_index()
    )
    return activity


def pivot_activity_table(activity: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot zone x heure avec colonnes multi-niveaux.
    """

    reset = activity.reset_index()
    return reset.pivot_table(
        index=activity.index.names[0],
        columns="hour",
        values=["trip_count", "tip_pct_mean"],
        aggfunc="mean",
    ).sort_index(axis=1)


def stack_activity_pivot(pivot_table: pd.DataFrame) -> pd.DataFrame:
    return pivot_table.stack(level=-1, future_stack=True)


def unstack_activity_table(activity: pd.DataFrame) -> pd.DataFrame:
    return activity.unstack("hour")


def swap_activity_levels(activity: pd.DataFrame) -> pd.DataFrame:
    return activity.swaplevel(0, 1).sort_index()


def top_tip_pct_pairs(
    activity: pd.DataFrame,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    return activity.sort_values("tip_pct_mean", ascending=False).head(top_n)


def load_raw_with_dask(
    pattern: str,
    *,
    columns: list[str] | None = None,
) -> dd.DataFrame:
    return dd.read_parquet(pattern, columns=columns)


def build_activity_dashboard_dask(
    ddf: dd.DataFrame,
    *,
    pickup_col: str = "PULocationID",
    datetime_col: str = "tpep_pickup_datetime",
    revenue_col: str = "total_amount",
    tip_pct_col: str = "tip_pct",
) -> dd.DataFrame:
    enriched = ddf.assign(
        hour=dd.to_datetime(ddf[datetime_col]).dt.hour.astype("int8"),
        weekday=dd.to_datetime(ddf[datetime_col]).dt.weekday.astype("int8"),
    )

    return (
        enriched.groupby([pickup_col, "hour", "weekday"])
        .agg(
            {
                pickup_col: "size",
                revenue_col: "sum",
                tip_pct_col: "mean",
            }
        )
        .rename(
            columns={
                pickup_col: "trip_count",
                revenue_col: "revenue_total",
                tip_pct_col: "tip_pct_mean",
            }
        )
    )


@dataclass
class BayesianTargetEncoder:
    smoothing: float = 10.0
    target_col: str | None = None
    keep_original: bool = True

    def __post_init__(self) -> None:
        self.mapping_: dict[Any, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, df: pd.DataFrame, column: str, target: str | None = None) -> "BayesianTargetEncoder":
        target_col = target or self.target_col
        if target_col is None:
            raise ValueError("Le nom de la cible est requis pour le target encoding.")
        if column not in df.columns or target_col not in df.columns:
            raise KeyError(f"Colonnes requises absentes: {column!r}, {target_col!r}.")

        self.global_mean_ = float(df[target_col].mean())
        stats = df.groupby(column, observed=False)[target_col].agg(["mean", "count"])
        smoothed = (
            (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_)
            / (stats["count"] + self.smoothing)
        )
        self.mapping_ = smoothed.to_dict()
        return self

    def transform(
        self,
        df: pd.DataFrame,
        column: str,
        *,
        encoded_col: str | None = None,
    ) -> pd.DataFrame:
        if not self.mapping_:
            raise RuntimeError("Le target encoder doit être entraîné avant transform().")
        if column not in df.columns:
            raise KeyError(f"Colonne absente: {column!r}.")

        encoded_col = encoded_col or f"{column}_target_enc"
        result = df.copy()
        result[encoded_col] = result[column].map(self.mapping_).fillna(self.global_mean_)
        if not self.keep_original:
            result = result.drop(columns=[column])
        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        column: str,
        target: str | None = None,
        *,
        encoded_col: str | None = None,
    ) -> pd.DataFrame:
        return self.fit(df, column, target=target).transform(df, column, encoded_col=encoded_col)


@dataclass
class CrossFittedTargetEncoder:
    smoothing: float = 10.0
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    target_col: str | None = None
    keep_original: bool = True

    def __post_init__(self) -> None:
        self.base_encoder_ = BayesianTargetEncoder(
            smoothing=self.smoothing,
            target_col=self.target_col,
            keep_original=self.keep_original,
        )

    def fit(self, df: pd.DataFrame, column: str, target: str | None = None) -> "CrossFittedTargetEncoder":
        self.base_encoder_.fit(df, column, target=target)
        return self

    def transform(
        self,
        df: pd.DataFrame,
        column: str,
        *,
        encoded_col: str | None = None,
    ) -> pd.DataFrame:
        return self.base_encoder_.transform(df, column, encoded_col=encoded_col)

    def fit_transform(
        self,
        df: pd.DataFrame,
        column: str,
        target: str | None = None,
        *,
        encoded_col: str | None = None,
    ) -> pd.DataFrame:
        target_col = target or self.target_col
        if target_col is None:
            raise ValueError("Le nom de la cible est requis pour le cross-fitted target encoding.")
        if self.n_splits < 2:
            raise ValueError("n_splits doit être >= 2.")

        encoded_col = encoded_col or f"{column}_target_enc"
        result = df.copy()
        encoded_values = pd.Series(index=result.index, dtype="float64")

        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        for train_idx, valid_idx in kfold.split(result):
            train_df = result.iloc[train_idx]
            valid_df = result.iloc[valid_idx]
            encoder = BayesianTargetEncoder(
                smoothing=self.smoothing,
                target_col=target_col,
                keep_original=True,
            ).fit(train_df, column, target=target_col)

            transformed = encoder.transform(valid_df, column, encoded_col=encoded_col)
            encoded_values.iloc[valid_idx] = transformed[encoded_col].to_numpy()

        result[encoded_col] = encoded_values
        if not self.keep_original:
            result = result.drop(columns=[column])

        self.fit(df, column, target=target_col)
        return result
