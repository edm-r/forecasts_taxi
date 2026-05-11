"""
src/models/registry.py
----------------------
Helpers MLflow Model Registry séparés du module d'entraînement.
"""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from models.train import promote_registered_model_version, register_model_from_run


def load_model_from_stage(
    model_name: str = "TipPredictor",
    stage: str = "Production",
) -> Any:
    """
    Charge un modèle depuis le registry MLflow.
    """

    return mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")


def list_model_versions(model_name: str = "TipPredictor") -> list[dict[str, str]]:
    """
    Retourne les versions connues d'un modèle registré.
    """

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    return [
        {
            "version": str(version.version),
            "current_stage": str(version.current_stage),
            "source": str(version.source),
            "run_id": str(version.run_id),
        }
        for version in versions
    ]


__all__ = [
    "load_model_from_stage",
    "list_model_versions",
    "promote_registered_model_version",
    "register_model_from_run",
]
