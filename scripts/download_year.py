"""
Script CLI pour télécharger les fichiers mensuels NYC Taxi.

Exemples :
    PYTHONPATH=src .venv/bin/python scripts/download_year.py --year 2023
    PYTHONPATH=src .venv/bin/python scripts/download_year.py --year 2023 --months 2 3 4
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from optim.async_io import _is_valid_parquet_file, download_year


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Téléchargement asynchrone des données NYC Taxi.")
    parser.add_argument("--year", type=int, default=2023, help="Année à télécharger.")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=ROOT / "data" / "raw",
        help="Répertoire de destination.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Nombre maximum de téléchargements concurrents.",
    )
    parser.add_argument(
        "--months",
        type=int,
        nargs="*",
        default=list(range(1, 13)),
        help="Liste explicite des mois à télécharger.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retélécharge même les fichiers déjà présents.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    dest_dir = args.dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Starting download year={args.year} months={args.months} "
        f"dest={dest_dir} concurrency={args.concurrency}",
        flush=True,
    )

    before_existing = {
        month: _is_valid_parquet_file(dest_dir / f"yellow_tripdata_{args.year}-{month:02d}.parquet")
        for month in args.months
    }

    paths = await download_year(
        args.year,
        dest_dir,
        concurrency=args.concurrency,
        months=args.months,
        skip_existing=not args.force,
    )

    for month, path in zip(args.months, paths):
        status = "skipped" if before_existing[month] and not args.force else "downloaded"
        print(f"{status:10s} {path}")


if __name__ == "__main__":
    asyncio.run(main())
