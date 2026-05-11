"""
scripts/run_loader.py
---------------------
Démo du LoaderFactory demandée à l'exercice 1.4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.loaders import LoaderFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Charge un fichier via LoaderFactory et affiche df.info()."
    )
    parser.add_argument("path", help="Chemin du fichier à charger.")
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Nombre de lignes à afficher après df.info().",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loader = LoaderFactory.create(args.path)
    df = loader.run(pd.DataFrame())
    df.info()
    print()
    print(df.head(args.head).to_string())


if __name__ == "__main__":
    main()
