"""
scripts/train_tip_model.py
--------------------------
Lance un entraînement local avec tracking MLflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from models.train import train_tip_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne le modèle de prédiction du tip_pct.")
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="Fichiers parquet de training. Par défaut: tous les mois 2023 du dossier raw.",
    )
    parser.add_argument(
        "--sample-rows-per-file",
        type=int,
        default=50_000,
        help="Nombre maximum de lignes échantillonnées par fichier.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Tracking URI MLflow. Exemple: http://localhost:5000 ou file:./mlruns",
    )
    parser.add_argument(
        "--experiment-name",
        default="tip_prediction",
        help="Nom de l'expérience MLflow.",
    )
    parser.add_argument(
        "--run-name",
        default="lgbm_v1",
        help="Nom du run MLflow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_tip_model(
        paths=[Path(path) for path in args.paths] if args.paths else None,
        sample_rows_per_file=args.sample_rows_per_file,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    print("Run ID:", result.run_id)
    print("Model URI:", result.model_uri)
    print("Backend:", result.model_backend)
    print("Metrics:", result.metrics)


if __name__ == "__main__":
    main()
