"""
src/api/schemas.py
------------------
Schémas Pydantic v2 pour les endpoints de prédiction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TripPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passenger_count: int = Field(ge=1, le=8)
    trip_distance: float = Field(gt=0, le=100)
    pickup_datetime: datetime
    pu_location_id: int = Field(ge=1, le=265)
    do_location_id: int = Field(ge=1, le=265)
    payment_type: Literal[1, 2, 3, 4]

    @model_validator(mode="after")
    def validate_trip_plausibility(self) -> "TripPredictionRequest":
        if self.trip_distance < 0.1:
            raise ValueError("trip_distance doit être >= 0.1 mile.")

        estimated_duration_minutes = self.trip_distance / 5.0 * 60.0
        if estimated_duration_minutes > 12 * 60:
            raise ValueError(
                "Course rejetée: durée plausible estimée supérieure à 12 heures."
            )
        return self


class TripPredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tip_pct_predicted: float = Field(ge=0.0, le=5.0)
    confidence_interval: tuple[float, float]
    model_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_interval(self) -> "TripPredictionResponse":
        low, high = self.confidence_interval
        if low > high:
            raise ValueError("confidence_interval doit être ordonné [low, high].")
        if not (low <= self.tip_pct_predicted <= high):
            raise ValueError("tip_pct_predicted doit être inclus dans confidence_interval.")
        return self


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predictions: list[TripPredictionResponse] = Field(min_length=1)
    model_version: str = Field(min_length=1)
