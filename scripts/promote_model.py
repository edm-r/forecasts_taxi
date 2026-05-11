"""
scripts/promote_model.py
------------------------
Enregistre un run MLflow dans le Model Registry puis le promeut de stage.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.train import promote_registered_model_version, register_model_from_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enregistre un run MLflow comme modèle puis le promeut de stage."
    )
    parser.add_argument("--run-id", required=True, help="Run ID MLflow source.")
    parser.add_argument(
        "--model-name",
        default="TipPredictor",
        help="Nom du modèle dans le registry.",
    )
    parser.add_argument(
        "--stage",
        default="Production",
        help="Stage cible après enregistrement.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="Tracking URI MLflow.",
    )
    parser.add_argument(
        "--artifact-path",
        default="model",
        help="Artifact path du modèle dans le run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri

    registered = register_model_from_run(
        args.run_id,
        model_name=args.model_name,
        artifact_path=args.artifact_path,
    )
    version = int(registered.version)
    promote_registered_model_version(
        args.model_name,
        version=version,
        stage=args.stage,
    )
    print("Model name:", args.model_name)
    print("Version:", version)
    print("Stage:", args.stage)


if __name__ == "__main__":
    main()
