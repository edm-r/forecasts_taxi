"""
src/models/train.py
-------------------
Entraînement tracké par MLflow et helpers de registry.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from core.config import Config
from models.features import MODEL_FEATURE_COLUMNS, RAW_TAXI_COLUMNS, prepare_training_frame

try:  # pragma: no cover - dépendance optionnelle
    from lightgbm import LGBMRegressor

    _HAS_LIGHTGBM = True
except (ImportError, OSError):  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingRegressor

    _HAS_LIGHTGBM = False


@dataclass
class TrainingResult:
    run_id: str
    experiment_name: str
    tracking_uri: str
    model_uri: str
    model_backend: str
    params: dict[str, Any]
    metrics: dict[str, float]
    feature_columns: list[str]


def load_training_dataframe(
    paths: Iterable[str | Path] | None = None,
    *,
    sample_rows_per_file: int | None = 50_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Charge un sous-ensemble reproductible des fichiers bruts.
    """

    resolved_paths = _resolve_training_paths(paths)
    rng = np.random.default_rng(random_state)
    frames: list[pd.DataFrame] = []
    effective_sample_rows = (
        None
        if sample_rows_per_file is None or sample_rows_per_file <= 0
        else sample_rows_per_file
    )

    for path in resolved_paths:
        df = pd.read_parquet(path, columns=RAW_TAXI_COLUMNS)
        if effective_sample_rows is not None and len(df) > effective_sample_rows:
            seed = int(rng.integers(0, 2**32 - 1))
            df = df.sample(n=effective_sample_rows, random_state=seed)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("Aucun fichier brut de training n'a été trouvé.")

    return pd.concat(frames, ignore_index=True)


def train_tip_model(
    *,
    paths: Iterable[str | Path] | None = None,
    sample_rows_per_file: int | None = 50_000,
    test_size: float = 0.2,
    random_state: int | None = None,
    tracking_uri: str | None = None,
    experiment_name: str = "tip_prediction",
    run_name: str = "lgbm_v1",
    model_params: dict[str, Any] | None = None,
) -> TrainingResult:
    """
    Entraîne un modèle de régression, logue les artefacts MLflow et renvoie
    les métadonnées du run.
    """

    config = Config()
    seed = int(random_state if random_state is not None else config.seed)
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    raw_df = load_training_dataframe(
        paths,
        sample_rows_per_file=sample_rows_per_file,
        random_state=seed,
    )
    X, y, cleaned = prepare_training_frame(raw_df)
    if len(X) < 20:
        raise ValueError("Pas assez de lignes valides pour entraîner le modèle.")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )

    params = _build_model_params(config, model_params)
    model, backend = _build_regressor(params, random_state=seed)

    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X_train, y_train)

        y_pred = np.asarray(model.predict(X_val), dtype="float64")
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae = float(mean_absolute_error(y_val, y_pred))
        residual_std = float(np.std(y_val.to_numpy(dtype="float64") - y_pred))

        signature = infer_signature(X_val, y_pred)

        log_params = {
            **params,
            "model_backend": backend,
            "n_train_rows": int(len(X_train)),
            "n_val_rows": int(len(X_val)),
            "n_input_features": int(X.shape[1]),
        }
        mlflow.log_params(log_params)
        mlflow.log_metrics(
            {
                "rmse_val": rmse,
                "mae_val": mae,
                "residual_std": residual_std,
            }
        )
        mlflow.log_dict({"feature_columns": MODEL_FEATURE_COLUMNS}, "feature_columns.json")
        mlflow.log_text(
            json.dumps({"cleaned_rows": int(len(cleaned)), "raw_rows": int(len(raw_df))}, indent=2),
            "training_summary.json",
        )

        figure_path = _log_feature_importance_figure(model, X_val, y_val, backend)

        _log_model_artifacts(
            model,
            backend=backend,
            signature=signature,
            input_example=X_val.head(3),
        )

        if figure_path is not None:
            mlflow.log_artifact(figure_path)
            Path(figure_path).unlink(missing_ok=True)

        run_id = run.info.run_id
        result = TrainingResult(
            run_id=run_id,
            experiment_name=experiment_name,
            tracking_uri=mlflow.get_tracking_uri(),
            model_uri=f"runs:/{run_id}/model",
            model_backend=backend,
            params=log_params,
            metrics={
                "rmse_val": rmse,
                "mae_val": mae,
                "residual_std": residual_std,
            },
            feature_columns=list(MODEL_FEATURE_COLUMNS),
        )

    return result


def register_model_from_run(
    run_id: str,
    *,
    model_name: str = "TipPredictor",
    artifact_path: str = "model",
) -> Any:
    """
    Enregistre un modèle MLflow à partir d'un run.
    """

    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.register_model(model_uri, model_name)


def promote_registered_model_version(
    model_name: str,
    *,
    version: int,
    stage: str = "Production",
    archive_existing_versions: bool = True,
) -> None:
    """
    Promeut une version du modèle dans le registry MLflow.
    """

    client = MlflowClient()
    client.transition_model_version_stage(
        model_name,
        version=version,
        stage=stage,
        archive_existing_versions=archive_existing_versions,
    )


def _resolve_training_paths(paths: Iterable[str | Path] | None) -> list[Path]:
    if paths is not None:
        return [Path(path) for path in paths]

    config = Config()
    raw_dir = config.raw_data_path
    return sorted(raw_dir.glob("yellow_tripdata_2023-*.parquet"))


def _build_model_params(config: Config, override: dict[str, Any] | None) -> dict[str, Any]:
    params = {
        "learning_rate": float(config.model["learning_rate"]),
        "n_estimators": int(config.model["n_estimators"]),
        "max_depth": int(config.model["max_depth"]),
        "objective": str(config.model["objective"]),
    }
    if override:
        params.update(override)
    return params


def _build_regressor(
    params: dict[str, Any],
    *,
    random_state: int,
) -> tuple[Any, str]:
    if _HAS_LIGHTGBM:
        model = LGBMRegressor(
            learning_rate=params["learning_rate"],
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            objective=params["objective"],
            random_state=random_state,
            n_jobs=-1,
        )
        return model, "lightgbm"

    model = HistGradientBoostingRegressor(
        learning_rate=params["learning_rate"],
        max_iter=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=random_state,
    )
    return model, "hist_gradient_boosting"


def _log_feature_importance_figure(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    backend: str,
) -> str | None:
    names = list(X_val.columns)
    values: np.ndarray

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype="float64")
    else:
        importance = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=3,
            random_state=42,
        )
        values = np.asarray(importance.importances_mean, dtype="float64")

    order = np.argsort(values)[::-1][:10]
    top_names = [names[idx] for idx in order]
    top_values = values[order]

    fd, tmp_path = tempfile.mkstemp(prefix=f"feature_importance_{backend}_", suffix=".png")
    Path(tmp_path).unlink(missing_ok=True)
    path = Path(tmp_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top_names[::-1], top_values[::-1], color="#2563eb")
    ax.set_title("Feature importance")
    ax.set_xlabel("importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _log_model_artifacts(
    model: Any,
    *,
    backend: str,
    signature: Any,
    input_example: pd.DataFrame,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "model"
        if backend == "lightgbm" and hasattr(mlflow, "lightgbm"):
            mlflow.lightgbm.save_model(
                model,
                path=str(model_dir),
                signature=signature,
                input_example=input_example,
            )
        else:
            mlflow.sklearn.save_model(
                model,
                path=str(model_dir),
                signature=signature,
                input_example=input_example,
            )
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
