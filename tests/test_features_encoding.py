"""
tests/test_features_encoding.py
-------------------------------
Tests pour l'encodage avancé et les tableaux d'activité.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.encoding import (
    BayesianTargetEncoder,
    CrossFittedTargetEncoder,
    build_activity_dashboard,
    build_activity_dashboard_dask,
    compute_streaming_statistics,
    iter_parquet_batches,
    load_raw_with_dask,
    optimize_dtypes,
    pivot_activity_table,
    stack_activity_pivot,
    summarize_dtype_optimization,
    swap_activity_levels,
    top_tip_pct_pairs,
    unstack_activity_table,
)


def sample_activity_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PULocationID": [1, 1, 2, 2, 2],
            "tpep_pickup_datetime": pd.to_datetime(
                [
                    "2023-01-01 08:00:00",
                    "2023-01-01 09:00:00",
                    "2023-01-02 08:00:00",
                    "2023-01-02 08:30:00",
                    "2023-01-02 09:00:00",
                ]
            ),
            "total_amount": [10.0, 20.0, 15.0, 10.0, 5.0],
            "tip_pct": [0.1, 0.2, 0.3, 0.1, 0.0],
            "zone_name": ["A", "A", "B", "B", "B"],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


class TestOptimizeDtypes:
    def test_optimize_dtypes_downcasts(self):
        df = pd.DataFrame(
            {
                "small_int": pd.Series([1, 2, 3], dtype="int64"),
                "small_float": pd.Series([1.0, 2.0, 3.0], dtype="float64"),
                "category_like": ["A", "A", "B"],
            }
        )
        optimized = optimize_dtypes(df)
        assert str(optimized["small_int"].dtype) != "int64"
        assert str(optimized["small_float"].dtype) in {"float32", "float64"}
        assert str(optimized["category_like"].dtype) == "category"

    def test_summarize_dtype_optimization(self):
        summary = summarize_dtype_optimization(sample_activity_df())
        assert summary["before_mb"] >= summary["after_mb"]


class TestStreamingParquet:
    def test_iter_parquet_batches_and_streaming_stats(self, tmp_path):
        df = sample_activity_df()
        path = tmp_path / "activity.parquet"
        df.to_parquet(path, index=False)

        batches = list(iter_parquet_batches(path, batch_size=2))
        assert [len(batch) for batch in batches] == [2, 2, 1]

        stats = compute_streaming_statistics(
            path,
            batch_size=2,
            fare_col="total_amount",
            tip_col="target",
        )
        assert stats["row_count"] == 5.0
        assert stats["fare_sum"] == pytest.approx(60.0)
        assert stats["tip_sum"] == pytest.approx(15.0)


class TestActivityDashboard:
    def test_build_activity_dashboard(self):
        activity = build_activity_dashboard(sample_activity_df())
        assert activity.index.names == ["PULocationID", "hour", "weekday"]
        assert "trip_count" in activity.columns

    def test_pivot_and_reshape(self):
        activity = build_activity_dashboard(sample_activity_df())
        pivot = pivot_activity_table(activity)
        assert isinstance(pivot.columns, pd.MultiIndex)
        assert not stack_activity_pivot(pivot).empty
        assert not unstack_activity_table(activity).empty
        assert swap_activity_levels(activity).index.names[0] == "hour"

    def test_top_tip_pct_pairs(self):
        activity = build_activity_dashboard(sample_activity_df())
        top = top_tip_pct_pairs(activity, top_n=2)
        assert len(top) == 2
        assert top["tip_pct_mean"].iloc[0] >= top["tip_pct_mean"].iloc[1]

    def test_dask_activity_matches_pandas(self, tmp_path):
        df = sample_activity_df()
        path = tmp_path / "activity.parquet"
        df.to_parquet(path, index=False)

        pandas_activity = build_activity_dashboard(df).sort_index()
        ddf = load_raw_with_dask(str(path))
        dask_activity = build_activity_dashboard_dask(ddf).compute().sort_index()
        pdt.assert_frame_equal(pandas_activity, dask_activity)


class TestTargetEncoders:
    def test_bayesian_target_encoder(self):
        df = sample_activity_df()
        encoder = BayesianTargetEncoder(smoothing=1.0, target_col="target")
        result = encoder.fit_transform(df, "PULocationID")
        assert "PULocationID_target_enc" in result.columns
        assert encoder.mapping_

    def test_cross_fitted_target_encoder(self):
        df = sample_activity_df()
        encoder = CrossFittedTargetEncoder(smoothing=1.0, n_splits=2, target_col="target")
        result = encoder.fit_transform(df, "PULocationID")
        assert "PULocationID_target_enc" in result.columns
        assert result["PULocationID_target_enc"].notna().all()

        transformed = encoder.transform(pd.DataFrame({"PULocationID": [1, 999]}), "PULocationID")
        assert transformed["PULocationID_target_enc"].notna().all()
