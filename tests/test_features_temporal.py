"""
tests/test_features_temporal.py
-------------------------------
Tests pour le feature engineering temporel.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.temporal import (
    TemporalFeaturizer,
    add_cyclical_features,
    aggregate_hourly_trip_counts,
    decompose_hourly_seasonality,
)


def sample_temporal_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(
                [
                    "2023-07-04 08:00:00",
                    "2023-07-04 18:30:00",
                    "2023-07-08 02:15:00",
                ]
            ),
            "tpep_dropoff_datetime": pd.to_datetime(
                [
                    "2023-07-04 08:20:00",
                    "2023-07-04 18:50:00",
                    "2023-07-08 02:35:00",
                ]
            ),
        }
    )


class TestTemporalFeaturizer:
    def test_adds_expected_columns(self):
        df = sample_temporal_df()
        result = TemporalFeaturizer().run(df)

        expected_cols = {
            "pickup_hour",
            "pickup_weekday",
            "pickup_day",
            "pickup_month",
            "pickup_quarter",
            "pickup_iso_week",
            "is_weekend",
            "is_us_holiday",
            "is_rush_hour",
            "is_late_night",
            "pickup_hour_sin",
            "pickup_hour_cos",
            "pickup_weekday_sin",
            "pickup_weekday_cos",
            "pickup_month_sin",
            "pickup_month_cos",
        }
        assert expected_cols.issubset(result.columns)

    def test_detects_holiday_and_proxy_flags(self):
        df = sample_temporal_df()
        result = TemporalFeaturizer().run(df)

        assert bool(result.loc[0, "is_us_holiday"]) is True
        assert bool(result.loc[0, "is_rush_hour"]) is True
        assert bool(result.loc[1, "is_rush_hour"]) is True
        assert bool(result.loc[2, "is_weekend"]) is True
        assert bool(result.loc[2, "is_late_night"]) is True

    def test_missing_datetime_column_raises(self):
        with pytest.raises(KeyError):
            TemporalFeaturizer(datetime_col="missing").run(sample_temporal_df())


class TestCyclicalFeatures:
    def test_add_cyclical_features(self):
        df = pd.DataFrame({"hour": [0, 6, 12, 18]})
        result = add_cyclical_features(df, columns={"hour": 24})
        assert np.isclose(result.loc[0, "hour_sin"], 0.0)
        assert np.isclose(result.loc[0, "hour_cos"], 1.0)


class TestHourlyCounts:
    def test_aggregate_hourly_trip_counts(self):
        df = sample_temporal_df()
        hourly = aggregate_hourly_trip_counts(df)
        assert hourly.sum() == 3
        assert hourly.index.is_monotonic_increasing

    def test_decompose_hourly_requires_statsmodels_or_enough_data(self):
        df = pd.DataFrame(
            {
                "tpep_pickup_datetime": pd.date_range("2023-01-01", periods=24, freq="h")
            }
        )
        if "statsmodels" not in sys.modules:
            with pytest.raises(RuntimeError):
                decompose_hourly_seasonality(df)
