"""
Script d'intégration de la partie 1.

Charge un mois de données via la Factory, orchestre le pipeline avec un
LoggingObserver, puis persiste le résultat dans data/processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.context import Timer
from core.metaclasses import BasePipelineStep
from pipeline.loaders import LoaderFactory
from pipeline.observers import LoggingObserver, MetricsObserver
from pipeline.orchestrator import Pipeline


class TipPctStep(BasePipelineStep):
    name = "tip_pct_step"

    def _run(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        safe_fare = result["fare_amount"].replace(0, np.nan)
        result["tip_pct"] = (result["tip_amount"] / safe_fare).fillna(0.0)
        return result


def main() -> None:
    raw_path = ROOT / "data" / "raw" / "yellow_tripdata_2023-01.parquet"
    output_dir = ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "yellow_tripdata_2023-01_part1.parquet"

    pipeline = Pipeline(
        [
            LoaderFactory.create(raw_path),
            TipPctStep(),
        ]
    )
    pipeline.attach_observer(LoggingObserver(ROOT / "pipeline.log"))
    metrics = MetricsObserver()
    pipeline.attach_observer(metrics)

    with Timer("Partie 1 pipeline"):
        df = pipeline.run(None)

    if df is None:
        raise RuntimeError("Le pipeline n'a produit aucun DataFrame.")

    df.to_parquet(output_path, index=False)
    print(f"Résultat écrit dans: {output_path}")
    print(f"Métriques collectées: {metrics.records}")


if __name__ == "__main__":
    main()
