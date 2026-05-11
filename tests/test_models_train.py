"""
tests/test_models_train.py
--------------------------
Tests du training MLflow et de la préparation des features.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.features import MODEL_FEATURE_COLUMNS, build_feature_matrix, prepare_training_frame
from models.train import train_tip_model


def sample_raw_df(rows: int = 80) -> pd.DataFrame:
    times = pd.date_range("2023-01-01 08:00:00", periods=rows, freq="h")
    fare_amount = pd.Series([10 + (idx % 7) for idx in range(rows)], dtype="float64")
    tip_amount = fare_amount * 0.18 + pd.Series([(idx % 3) * 0.02 for idx in range(rows)], dtype="float64")
    return pd.DataFrame(
        {
            "passenger_count": [(idx % 4) + 1 for idx in range(rows)],
            "trip_distance": [1.5 + (idx % 10) * 0.3 for idx in range(rows)],
            "tpep_pickup_datetime": times,
            "PULocationID": [(idx % 20) + 1 for idx in range(rows)],
            "DOLocationID": [((idx + 7) % 20) + 1 for idx in range(rows)],
            "payment_type": [(idx % 4) + 1 for idx in range(rows)],
            "fare_amount": fare_amount,
            "tip_amount": tip_amount,
        }
    )


class TestModelFeatures:
    def test_prepare_training_frame(self):
        X, y, cleaned = prepare_training_frame(sample_raw_df())
        assert list(X.columns) == MODEL_FEATURE_COLUMNS
        assert len(X) == len(y) == len(cleaned)
        assert X.isna().sum().sum() == 0

    def test_build_feature_matrix_from_canonical_input(self):
        raw = sample_raw_df(5).rename(
            columns={
                "tpep_pickup_datetime": "pickup_datetime",
                "PULocationID": "pu_location_id",
                "DOLocationID": "do_location_id",
            }
        )
        features = build_feature_matrix(
            raw[
                [
                    "passenger_count",
                    "trip_distance",
                    "pickup_datetime",
                    "pu_location_id",
                    "do_location_id",
                    "payment_type",
                ]
            ]
        )
        assert list(features.columns) == MODEL_FEATURE_COLUMNS


class TestTrainingMlflow:
    def test_train_tip_model(self, tmp_path):
        path = tmp_path / "train.parquet"
        sample_raw_df().to_parquet(path, index=False)
        tracking_dir = (tmp_path / "mlruns").resolve()
        tracking_uri = tracking_dir.as_uri()

        result = train_tip_model(
            paths=[Path(path)],
            sample_rows_per_file=None,
            test_size=0.25,
            random_state=42,
            tracking_uri=tracking_uri,
            experiment_name="pytest_tip_prediction",
            run_name="pytest_run",
            model_params={"n_estimators": 20, "max_depth": 3},
        )

        assert result.run_id
        assert result.model_uri.startswith("runs:/")
        assert result.metrics["rmse_val"] >= 0.0

        model = mlflow.pyfunc.load_model(result.model_uri)
        X, _, _ = prepare_training_frame(sample_raw_df(10))
        preds = model.predict(X.head(3))
        assert len(preds) == 3
