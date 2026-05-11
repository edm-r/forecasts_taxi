"""
tests/test_api_schemas.py
-------------------------
Tests des schémas Pydantic de l'API.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from api.schemas import TripPredictionRequest, TripPredictionResponse


def valid_payload() -> dict:
    return {
        "passenger_count": 2,
        "trip_distance": 3.4,
        "pickup_datetime": datetime(2023, 1, 5, 8, 30),
        "pu_location_id": 10,
        "do_location_id": 20,
        "payment_type": 1,
    }


class TestTripPredictionRequest:
    def test_valid_request(self):
        request = TripPredictionRequest(**valid_payload())
        assert request.passenger_count == 2

    def test_invalid_distance_rejected(self):
        payload = valid_payload()
        payload["trip_distance"] = 0.05
        with pytest.raises(ValidationError):
            TripPredictionRequest(**payload)

    def test_invalid_payment_type_rejected(self):
        payload = valid_payload()
        payload["payment_type"] = 7
        with pytest.raises(ValidationError):
            TripPredictionRequest(**payload)


class TestTripPredictionResponse:
    def test_valid_response(self):
        response = TripPredictionResponse(
            tip_pct_predicted=0.2,
            confidence_interval=(0.1, 0.3),
            model_version="TipPredictor/Production",
        )
        assert response.tip_pct_predicted == 0.2

    def test_interval_must_contain_prediction(self):
        with pytest.raises(ValidationError):
            TripPredictionResponse(
                tip_pct_predicted=0.5,
                confidence_interval=(0.1, 0.3),
                model_version="TipPredictor/Production",
            )
