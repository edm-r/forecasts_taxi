"""
tests/test_features_windowing.py
--------------------------------
Tests pour les window functions et le time-series engineering.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import features.windowing as windowing
from features.windowing import (
    add_expanding_tip_pct_mean,
    add_fare_rank_pct,
    add_fourier_power_features,
    add_timeseries_differences,
    add_timeseries_lag_features,
    add_timeseries_rolling_features,
    add_zone_previous_tip_mean,
    aggregate_hourly_zone_timeseries,
    compute_zone_daily_trip_lags,
    trimmed_mean,
)


def sample_trip_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PULocationID": [1, 1, 1, 2, 2, 1],
            "tpep_pickup_datetime": pd.to_datetime(
                [
                    "2023-01-01 08:00:00",
                    "2023-01-01 09:00:00",
                    "2023-01-01 10:00:00",
                    "2023-01-01 08:15:00",
                    "2023-01-02 09:00:00",
                    "2023-01-04 08:00:00",
                ]
            ),
            "tip_amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "fare_amount": [10.0, 20.0, 15.0, 8.0, 12.0, 30.0],
            "tip_pct": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )


def sample_hourly_ts() -> pd.DataFrame:
    hours = pd.date_range("2023-01-01", periods=8, freq="h")
    return pd.DataFrame(
        {
            "PULocationID": [1] * 8,
            "pickup_hour": hours,
            "trip_count": [10, 20, 30, 40, 10, 20, 30, 40],
            "tip_mean": [1.0, 1.1, 1.2, 1.3, 1.0, 1.1, 1.2, 1.3],
            "fare_mean": [5.0, 5.5, 6.0, 6.5, 5.0, 5.5, 6.0, 6.5],
        }
    )


class TestWindowFeatures:
    def test_add_zone_previous_tip_mean(self):
        result = add_zone_previous_tip_mean(sample_trip_df(), window=2)
        zone1 = (
            result[result["PULocationID"] == 1]
            .sort_values("tpep_pickup_datetime")
            ["tip_amount_prev_5_mean"]
            .tolist()
        )
        assert np.isnan(zone1[0])
        assert zone1[1] == pytest.approx(1.0)
        assert zone1[2] == pytest.approx(1.5)
        assert zone1[3] == pytest.approx(2.5)

    def test_add_fare_rank_pct(self):
        result = add_fare_rank_pct(sample_trip_df())
        day1_zone1 = (
            result[
                (result["PULocationID"] == 1)
                & (result["tpep_pickup_datetime"].dt.normalize() == pd.Timestamp("2023-01-01"))
            ]
            .sort_values("fare_amount")
            ["fare_amount_rank_pct"]
            .tolist()
        )
        assert day1_zone1 == pytest.approx([1 / 3, 2 / 3, 1.0])

    def test_add_expanding_tip_pct_mean(self):
        result = add_expanding_tip_pct_mean(sample_trip_df())
        zone1 = (
            result[result["PULocationID"] == 1]
            .sort_values("tpep_pickup_datetime")
            ["tip_pct_expanding_mean"]
            .tolist()
        )
        assert zone1 == pytest.approx([0.1, 0.15, 0.2, 0.3])

    def test_compute_zone_daily_trip_lags(self):
        result = compute_zone_daily_trip_lags(sample_trip_df(), lags=(1, 3))
        zone1 = result[result["PULocationID"] == 1].set_index("service_day")
        assert zone1.loc[pd.Timestamp("2023-01-03"), "trip_count"] == 0
        assert zone1.loc[pd.Timestamp("2023-01-04"), "trip_count_lag_1d"] == 0
        assert zone1.loc[pd.Timestamp("2023-01-04"), "trip_count_lag_3d"] == 3

    def test_trimmed_mean(self):
        series = pd.Series([1, 2, 3, 100, 200], dtype="float64")
        assert trimmed_mean(series, proportion=0.2) == pytest.approx((2 + 3 + 100) / 3)


class TestTimeSeriesEngineering:
    def test_aggregate_hourly_zone_timeseries(self):
        result = aggregate_hourly_zone_timeseries(sample_trip_df())
        assert {"trip_count", "tip_mean", "fare_mean"}.issubset(result.columns)
        assert result["trip_count"].sum() == len(sample_trip_df())

    def test_add_timeseries_lag_features(self):
        result = add_timeseries_lag_features(sample_hourly_ts(), lags=(1, 2))
        assert np.isnan(result.loc[0, "trip_count_lag_1"])
        assert result.loc[1, "trip_count_lag_1"] == 10
        assert result.loc[2, "trip_count_lag_2"] == 10

    def test_add_timeseries_rolling_features(self):
        result = add_timeseries_rolling_features(sample_hourly_ts(), windows=(2,))
        assert np.isnan(result.loc[0, "trip_count_rolling_mean_2"])
        assert result.loc[1, "trip_count_rolling_mean_2"] == pytest.approx(10.0)
        assert result.loc[2, "trip_count_rolling_mean_2"] == pytest.approx(15.0)
        assert result.loc[2, "trip_count_rolling_max_2"] == pytest.approx(20.0)

    def test_add_timeseries_differences(self):
        result = add_timeseries_differences(sample_hourly_ts(), periods=(1,))
        assert np.isnan(result.loc[0, "trip_count_diff_1"])
        assert result.loc[1, "trip_count_diff_1"] == 10

    def test_add_fourier_power_features(self):
        result = add_fourier_power_features(
            sample_hourly_ts(),
            history_hours=4,
            top_k=2,
        )
        assert result["trip_count_fourier_power_1"].notna().sum() == 5
        assert result["trip_count_fourier_power_2"].notna().sum() == 5

    def test_adf_stationarity_report(self):
        series = pd.Series(np.random.default_rng(42).normal(size=240))
        if windowing.adfuller is None:
            with pytest.raises(RuntimeError):
                windowing.adf_stationarity_report(series)
        else:
            report = windowing.adf_stationarity_report(series)
            assert {"statistic", "pvalue", "usedlag", "nobs"}.issubset(report)
