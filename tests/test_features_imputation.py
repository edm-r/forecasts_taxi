"""
tests/test_features_imputation.py
---------------------------------
Tests pour l'imputation avancée.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import features.imputation as imputation
from features.imputation import (
    compare_imputation_strategies,
    evaluate_imputation_rmse,
    introduce_missingness,
    knn_impute,
    median_impute,
    mice_impute,
    multiple_mice_imputations,
    plot_missingness,
    rubin_pool,
    summarize_missingness,
)


def sample_imputation_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "passenger_count": [1.0, 2.0, 1.0, 3.0, 2.0, 4.0],
            "trip_distance": [1.2, 2.4, 1.8, 3.1, 2.9, 4.2],
            "fare_amount": [10.0, 15.0, 12.0, 20.0, 18.0, 25.0],
            "vendor": ["A", "A", "B", "B", "A", "B"],
        }
    )


class TestMissingness:
    def test_summarize_missingness(self):
        df = sample_imputation_df()
        df.loc[[1, 3], "trip_distance"] = np.nan
        summary = summarize_missingness(df)
        assert summary.loc["trip_distance", "missing_count"] == 2
        assert summary.loc["trip_distance", "missing_ratio"] == pytest.approx(2 / 6)

    def test_plot_missingness_behaviour(self):
        if imputation.msno is None:
            with pytest.raises(RuntimeError):
                plot_missingness(sample_imputation_df())

    def test_introduce_missingness(self):
        masked, masks = introduce_missingness(
            sample_imputation_df(),
            columns=["passenger_count", "trip_distance"],
            fraction=1 / 3,
            random_state=42,
        )
        assert masked["passenger_count"].isna().sum() == 2
        assert masked["trip_distance"].isna().sum() == 2
        assert set(masks) == {"passenger_count", "trip_distance"}


class TestImputers:
    def test_median_impute(self):
        df = sample_imputation_df()
        df.loc[[0, 1], "passenger_count"] = np.nan
        result = median_impute(df, columns=["passenger_count"])
        assert result["passenger_count"].isna().sum() == 0

    def test_knn_and_mice_impute(self):
        df = sample_imputation_df()
        df.loc[[0, 4], "trip_distance"] = np.nan
        knn = knn_impute(df, columns=["trip_distance"], n_neighbors=2)
        mice = mice_impute(df, columns=["trip_distance"], random_state=42, max_iter=5)
        assert knn["trip_distance"].isna().sum() == 0
        assert mice["trip_distance"].isna().sum() == 0

    def test_multiple_mice_imputations(self):
        df = sample_imputation_df()
        df.loc[[1, 5], "passenger_count"] = np.nan
        imputations = multiple_mice_imputations(
            df,
            columns=["passenger_count"],
            n_imputations=3,
            random_state=42,
            max_iter=5,
        )
        assert len(imputations) == 3
        assert all(imputed["passenger_count"].isna().sum() == 0 for imputed in imputations)

    def test_evaluate_imputation_rmse(self):
        original = sample_imputation_df()
        imputed = original.copy()
        masks = {"trip_distance": pd.Index([0, 2])}
        score = evaluate_imputation_rmse(original, imputed, masks=masks)
        assert score["trip_distance"] == 0.0

    def test_compare_imputation_strategies(self):
        comparison = compare_imputation_strategies(
            sample_imputation_df(),
            columns=["passenger_count", "trip_distance"],
            fraction=1 / 3,
            random_state=42,
            n_neighbors=2,
        )
        assert {"median_rmse", "knn_rmse", "mice_rmse", "multiple_mice_rmse"}.issubset(comparison)
        assert len(comparison["multiple_mice_rmse"]) == 5


class TestRubinPooling:
    def test_rubin_pool(self):
        pooled = rubin_pool(
            [
                np.array([1.0, 2.0, 3.0]),
                np.array([2.0, 3.0, 4.0]),
                np.array([3.0, 4.0, 5.0]),
            ]
        )
        assert np.allclose(pooled["pooled_mean"], np.array([2.0, 3.0, 4.0]))
        assert pooled["within_var"].shape == (3,)
        assert pooled["between_var"].shape == (3,)
        assert pooled["total_var"].shape == (3,)
