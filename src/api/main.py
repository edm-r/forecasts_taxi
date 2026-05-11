"""
src/api/main.py
---------------
API REST FastAPI pour servir le modèle de prédiction du tip_pct.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Callable

import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, Request
from mlflow.tracking import MlflowClient
from prometheus_client import CollectorRegistry
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import (
    BatchPredictionResponse,
    TripPredictionRequest,
    TripPredictionResponse,
)
from models.features import build_feature_matrix, frame_from_requests


ModelLoader = Callable[[str], Any]


def create_app(
    *,
    model_uri: str | None = None,
    model_loader: ModelLoader | None = None,
    metrics_registry: CollectorRegistry | None = None,
) -> FastAPI:
    resolved_model_uri = model_uri or os.getenv(
        "MLFLOW_MODEL_URI",
        "models:/TipPredictor/Production",
    )
    resolved_loader = model_loader or mlflow.pyfunc.load_model

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model = resolved_loader(resolved_model_uri)
        app.state.model = model
        app.state.model_uri = resolved_model_uri
        app.state.model_version = _resolve_model_version(resolved_model_uri)
        app.state.residual_std = _resolve_residual_std(model)
        yield

    app = FastAPI(
        title="Tip Predictor API",
        version="1.0.0",
        lifespan=lifespan,
    )

    _instrument_app(app, metrics_registry=metrics_registry)

    @app.get("/health")
    async def health(request: Request) -> dict[str, str]:
        return {
            "status": "ok",
            "model_version": request.app.state.model_version,
        }

    @app.post("/predict", response_model=TripPredictionResponse)
    async def predict(payload: TripPredictionRequest, request: Request) -> TripPredictionResponse:
        response = _predict_single(
            request.app.state.model,
            payload.model_dump(),
            model_version=request.app.state.model_version,
            residual_std=request.app.state.residual_std,
        )
        return TripPredictionResponse(**response)

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(
        payload: list[TripPredictionRequest],
        request: Request,
    ) -> BatchPredictionResponse:
        records = [trip.model_dump() for trip in payload]
        predictions = _predict_many(
            request.app.state.model,
            records,
            model_version=request.app.state.model_version,
            residual_std=request.app.state.residual_std,
        )
        return BatchPredictionResponse(
            predictions=[TripPredictionResponse(**item) for item in predictions],
            model_version=request.app.state.model_version,
        )

    return app


def _instrument_app(
    app: FastAPI,
    *,
    metrics_registry: CollectorRegistry | None,
) -> None:
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=False,
        should_group_untemplated=True,
        excluded_handlers=[
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
        ],
        registry=metrics_registry,
    )
    instrumentator.instrument(app).expose(
        app,
        endpoint="/metrics",
        include_in_schema=False,
        should_gzip=True,
    )


def _predict_single(
    model: Any,
    record: dict[str, Any],
    *,
    model_version: str,
    residual_std: float,
) -> dict[str, Any]:
    return _predict_many(
        model,
        [record],
        model_version=model_version,
        residual_std=residual_std,
    )[0]


def _predict_many(
    model: Any,
    records: list[dict[str, Any]],
    *,
    model_version: str,
    residual_std: float,
) -> list[dict[str, Any]]:
    frame = frame_from_requests(records)
    features = build_feature_matrix(frame)
    predictions = np.asarray(model.predict(features), dtype="float64")

    responses: list[dict[str, Any]] = []
    for pred in predictions:
        clipped_pred = float(np.clip(pred, 0.0, 5.0))
        low, high = _confidence_interval(clipped_pred, residual_std)
        responses.append(
            {
                "tip_pct_predicted": clipped_pred,
                "confidence_interval": (low, high),
                "model_version": model_version,
            }
        )
    return responses


def _confidence_interval(prediction: float, residual_std: float) -> tuple[float, float]:
    spread = max(0.05, residual_std * 1.96)
    low = float(max(0.0, prediction - spread))
    high = float(min(5.0, prediction + spread))
    if low > prediction:
        low = prediction
    if high < prediction:
        high = prediction
    return low, high


def _resolve_residual_std(model: Any) -> float:
    metadata = getattr(model, "metadata", None)
    run_id = getattr(metadata, "run_id", None)
    if run_id:
        client = MlflowClient()
        run = client.get_run(run_id)
        metric = run.data.metrics.get("residual_std")
        if metric is not None:
            return float(metric)
    return 0.05


def _resolve_model_version(model_uri: str) -> str:
    if model_uri.startswith("models:/"):
        return model_uri.split("models:/", 1)[1]
    return model_uri


app = create_app()
