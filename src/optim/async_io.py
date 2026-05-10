"""
src/optim/async_io.py
---------------------
Téléchargement asynchrone des données NYC Taxi.

Partie 2 — Exercice 2.5
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

import aiohttp


def build_month_url(year: int, month: int) -> str:
    return (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        f"yellow_tripdata_{year}-{month:02d}.parquet"
    )


async def download_month(
    session: aiohttp.ClientSession,
    year: int,
    month: int,
    dest_dir: str | Path,
    *,
    semaphore: asyncio.Semaphore | None = None,
    chunk_size: int = 1 << 20,
    retries: int = 3,
) -> Path:
    """
    Télécharge un mois de données au format Parquet.
    """

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    url = build_month_url(year, month)
    output_path = dest_dir / f"yellow_tripdata_{year}-{month:02d}.parquet"

    async def _download() -> Path:
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    with output_path.open("wb") as fh:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            fh.write(chunk)
                return output_path
            except Exception as exc:  # pragma: no cover - la boucle est testée indirectement
                last_error = exc
                if attempt == retries:
                    raise
                await asyncio.sleep(0.2 * attempt)

        assert last_error is not None
        raise last_error

    if semaphore is None:
        return await _download()

    async with semaphore:
        return await _download()


async def download_year(
    year: int,
    dest_dir: str | Path,
    *,
    concurrency: int = 4,
    months: Iterable[int] | None = None,
    timeout_s: float = 60.0,
) -> list[Path]:
    """
    Télécharge plusieurs mois en concurrence avec limitation de flux.
    """

    month_list = list(months or range(1, 13))
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            download_month(session, year, month, dest_dir, semaphore=semaphore)
            for month in month_list
        ]
        return list(await asyncio.gather(*tasks))


def download_year_sync(
    year: int,
    dest_dir: str | Path,
    *,
    months: Iterable[int] | None = None,
    chunk_size: int = 1 << 20,
) -> list[Path]:
    """
    Version synchrone simple pour comparaison de benchmark.
    """

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for month in months or range(1, 13):
        url = build_month_url(year, month)
        output_path = dest_dir / f"yellow_tripdata_{year}-{month:02d}.parquet"
        with urlopen(url) as response, output_path.open("wb") as fh:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
        paths.append(output_path)

    return paths
