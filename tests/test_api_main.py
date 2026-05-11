"""
tests/test_api_main.py
----------------------
Tests des endpoints FastAPI.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api.main import create_app


class DummyModel:
    def predict(self, frame):
        return np.full(len(frame), 0.2, dtype="float64")


def valid_payload() -> dict:
    return {
        "passenger_count": 2,
        "trip_distance": 3.4,
        "pickup_datetime": "2023-01-05T08:30:00",
        "pu_location_id": 10,
        "do_location_id": 20,
        "payment_type": 1,
    }


class TestApi:
    def test_endpoints(self):
        calls = {"count": 0}

        def loader(model_uri: str):
            calls["count"] += 1
            assert model_uri == "models:/TipPredictor/Production"
            return DummyModel()

        app = create_app(
            model_uri="models:/TipPredictor/Production",
            model_loader=loader,
            metrics_registry=CollectorRegistry(),
        )

        with TestClient(app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            pred = client.post("/predict", json=valid_payload())
            assert pred.status_code == 200
            assert pred.json()["tip_pct_predicted"] == 0.2

            batch = client.post("/predict/batch", json=[valid_payload(), valid_payload()])
            assert batch.status_code == 200
            assert len(batch.json()["predictions"]) == 2

            metrics = client.get("/metrics")
            assert metrics.status_code == 200
            assert "http_requests_total" in metrics.text
            assert "http_request_duration_highr_seconds" in metrics.text

        assert calls["count"] == 1

    def test_validation_error(self):
        app = create_app(
            model_uri="models:/TipPredictor/Production",
            model_loader=lambda _: DummyModel(),
            metrics_registry=CollectorRegistry(),
        )
        with TestClient(app) as client:
            payload = valid_payload()
            payload["trip_distance"] = 0.01
            response = client.post("/predict", json=payload)
            assert response.status_code == 422
