"""
tests/test_optim_vectorize.py
-----------------------------
Tests pour les calculs naïfs et vectorisés.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pandas.testing as pdt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optim.vectorize import (
    apply_dtype_recommendations,
    build_centroid_distance_matrix,
    compute_features_naive,
    compute_features_vectorized,
    recommend_optimized_dtypes,
    summarize_memory_optimization,
)


def make_trip_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                ["2023-01-01 08:00:00", "2023-01-01 09:15:00", "2023-01-01 10:00:00"]
            ),
            "tpep_dropoff_datetime": pd.to_datetime(
                ["2023-01-01 08:30:00", "2023-01-01 09:15:00", "2023-01-01 10:20:00"]
            ),
            "trip_distance": [10.0, 3.0, 5.5],
            "fare_amount": [20.0, 15.0, 12.5],
        }
    )


class TestComputeFeatures:
    def test_naive_and_vectorized_match(self):
        df = make_trip_df()

        naive = compute_features_naive(df)
        vectorized = compute_features_vectorized(df)

        pdt.assert_frame_equal(naive, vectorized)

    def test_zero_duration_produces_zero_speed(self):
        df = make_trip_df()
        result = compute_features_vectorized(df)
        assert result.loc[1, "avg_speed_mph"] == 0.0

    def test_invalid_fare_amount_produces_nan_penalty(self):
        df = make_trip_df()
        df.loc[0, "fare_amount"] = -2.0

        naive = compute_features_naive(df)
        vectorized = compute_features_vectorized(df)

        assert np.isnan(naive.loc[0, "penalty"])
        assert np.isnan(vectorized.loc[0, "penalty"])
        pdt.assert_frame_equal(naive, vectorized)


class TestDistanceMatrix:
    def test_build_centroid_distance_matrix(self):
        centroids = pd.DataFrame({"x": [0.0, 3.0, 0.0], "y": [0.0, 4.0, 4.0]})

        matrix = build_centroid_distance_matrix(centroids)

        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 0.0)
        assert np.allclose(matrix, matrix.T)
        assert np.isclose(matrix[0, 1], 5.0)


class TestDtypeRecommendations:
    def test_recommend_optimized_dtypes(self):
        df = pd.DataFrame(
            {
                "small_int": pd.Series([1, 2, 3], dtype="int64"),
                "small_float": pd.Series([1.0, 2.5, 3.5], dtype="float64"),
                "city": ["A", "A", "B"],
            }
        )

        recommendations = recommend_optimized_dtypes(df)

        assert recommendations["small_int"] == "int8"
        assert recommendations["small_float"] == "float32"
        assert recommendations["city"] == "category"

    def test_apply_dtype_recommendations(self):
        df = pd.DataFrame({"value": pd.Series([1, 2, 3], dtype="int64")})
        optimized = apply_dtype_recommendations(df, {"value": "int8"})
        assert str(optimized["value"].dtype) == "int8"

    def test_summarize_memory_optimization(self):
        df = pd.DataFrame(
            {
                "value": pd.Series([1, 2, 3], dtype="int64"),
                "city": ["A", "A", "B"],
            }
        )

        summary = summarize_memory_optimization(df)

        assert "before_mb" in summary
        assert "after_mb" in summary
        assert isinstance(summary["recommendations"], dict)
