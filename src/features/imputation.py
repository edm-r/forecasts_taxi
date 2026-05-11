"""
src/features/imputation.py
-------------------------
Imputation avancée et évaluation des stratégies de missing data.

Partie 3 — Exercice 3.7
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error

try:  # pragma: no cover - dépendance optionnelle
    import missingno as msno
except ImportError:  # pragma: no cover
    msno = None


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_ratio": df.isna().mean(),
        }
    )
    return summary.sort_values(["missing_count", "missing_ratio"], ascending=False)


def plot_missingness(df: pd.DataFrame) -> dict[str, Any]:
    if msno is None:
        raise RuntimeError(
            "missingno est requis pour la visualisation des valeurs manquantes."
        )

    return {
        "matrix_ax": msno.matrix(df),
        "heatmap_ax": msno.heatmap(df),
    }


def introduce_missingness(
    df: pd.DataFrame,
    *,
    columns: list[str],
    fraction: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, pd.Index]]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction doit être dans [0, 1].")

    rng = np.random.default_rng(random_state)
    masked = df.copy()
    masks: dict[str, pd.Index] = {}

    for column in columns:
        if column not in masked.columns:
            raise KeyError(f"Colonne absente: {column!r}.")

        available = masked.index[masked[column].notna()]
        sample_size = int(round(len(available) * fraction))
        chosen = available[:0]
        if sample_size > 0:
            chosen = pd.Index(rng.choice(available.to_numpy(), size=sample_size, replace=False))
            masked.loc[chosen, column] = np.nan
        masks[column] = chosen

    return masked, masks


def median_impute(df: pd.DataFrame, *, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        median = result[column].median()
        result[column] = result[column].fillna(median)
    return result


def knn_impute(
    df: pd.DataFrame,
    *,
    columns: list[str],
    n_neighbors: int = 5,
) -> pd.DataFrame:
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    result[numeric_cols] = imputer.fit_transform(result[numeric_cols])
    return result


def mice_impute(
    df: pd.DataFrame,
    *,
    columns: list[str],
    random_state: int = 42,
    max_iter: int = 10,
) -> pd.DataFrame:
    result = df.copy()
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        random_state=random_state,
        max_iter=max_iter,
        sample_posterior=False,
    )
    result[numeric_cols] = imputer.fit_transform(result[numeric_cols])
    return result


def multiple_mice_imputations(
    df: pd.DataFrame,
    *,
    columns: list[str],
    n_imputations: int = 5,
    random_state: int = 42,
    max_iter: int = 10,
) -> list[pd.DataFrame]:
    imputations: list[pd.DataFrame] = []
    for offset in range(n_imputations):
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            random_state=random_state + offset,
            max_iter=max_iter,
            sample_posterior=True,
        )
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        result[numeric_cols] = imputer.fit_transform(result[numeric_cols])
        imputations.append(result)
    return imputations


def evaluate_imputation_rmse(
    original_df: pd.DataFrame,
    imputed_df: pd.DataFrame,
    *,
    masks: dict[str, pd.Index],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for column, index in masks.items():
        if len(index) == 0:
            scores[column] = 0.0
            continue
        truth = original_df.loc[index, column].to_numpy(dtype="float64")
        pred = imputed_df.loc[index, column].to_numpy(dtype="float64")
        scores[column] = float(np.sqrt(mean_squared_error(truth, pred)))
    return scores


def compare_imputation_strategies(
    df: pd.DataFrame,
    *,
    columns: list[str],
    fraction: float = 0.05,
    random_state: int = 42,
    n_neighbors: int = 5,
) -> dict[str, Any]:
    masked_df, masks = introduce_missingness(
        df,
        columns=columns,
        fraction=fraction,
        random_state=random_state,
    )

    median_df = median_impute(masked_df, columns=columns)
    knn_df = knn_impute(masked_df, columns=columns, n_neighbors=n_neighbors)
    mice_df = mice_impute(masked_df, columns=columns, random_state=random_state)
    multiple = multiple_mice_imputations(masked_df, columns=columns, random_state=random_state)

    return {
        "masked_df": masked_df,
        "masks": masks,
        "median_rmse": evaluate_imputation_rmse(df, median_df, masks=masks),
        "knn_rmse": evaluate_imputation_rmse(df, knn_df, masks=masks),
        "mice_rmse": evaluate_imputation_rmse(df, mice_df, masks=masks),
        "multiple_mice_rmse": [
            evaluate_imputation_rmse(df, imputed_df, masks=masks)
            for imputed_df in multiple
        ],
    }


def rubin_pool(predictions: list[np.ndarray]) -> dict[str, np.ndarray]:
    if not predictions:
        raise ValueError("Au moins un jeu de prédictions est requis.")

    stacked = np.vstack(predictions)
    pooled_mean = stacked.mean(axis=0)

    # Avec uniquement des jeux de prédictions, on ne dispose pas de la variance
    # intra-imputation propre à chaque modèle. On la laisse donc à zéro et on
    # propage uniquement la variance inter-imputations.
    within_var = np.zeros(stacked.shape[1], dtype="float64")
    between_var = (
        stacked.var(axis=0, ddof=1)
        if stacked.shape[0] > 1
        else np.zeros(stacked.shape[1], dtype="float64")
    )
    total_var = within_var + (1.0 + 1.0 / stacked.shape[0]) * between_var

    return {
        "pooled_mean": pooled_mean,
        "within_var": within_var,
        "between_var": between_var,
        "total_var": total_var,
    }
