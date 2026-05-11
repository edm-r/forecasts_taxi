"""
tests/test_optim_async_io.py
----------------------------
Tests pour le téléchargement asynchrone.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optim import async_io


@pytest.fixture
def http_file_server(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    for month in (1, 2):
        (source_dir / f"yellow_tripdata_2023-{month:02d}.parquet").write_bytes(
            f"month-{month}".encode("utf-8")
        )

    handler = partial(SimpleHTTPRequestHandler, directory=str(source_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield source_dir, server.server_address[1]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class TestAsyncDownloads:
    def test_download_year_async(self, http_file_server, tmp_path, monkeypatch):
        _, port = http_file_server

        def fake_build_month_url(year: int, month: int) -> str:
            return f"http://127.0.0.1:{port}/yellow_tripdata_{year}-{month:02d}.parquet"

        monkeypatch.setattr(async_io, "build_month_url", fake_build_month_url)

        output_dir = tmp_path / "downloads_async"
        paths = asyncio.run(async_io.download_year(2023, output_dir, concurrency=2, months=[1, 2]))

        assert len(paths) == 2
        assert paths[0].read_bytes() == b"month-1"
        assert paths[1].read_bytes() == b"month-2"

    def test_download_year_sync(self, http_file_server, tmp_path, monkeypatch):
        _, port = http_file_server

        def fake_build_month_url(year: int, month: int) -> str:
            return f"http://127.0.0.1:{port}/yellow_tripdata_{year}-{month:02d}.parquet"

        monkeypatch.setattr(async_io, "build_month_url", fake_build_month_url)

        output_dir = tmp_path / "downloads_sync"
        paths = async_io.download_year_sync(2023, output_dir, months=[1, 2])

        assert len(paths) == 2
        assert paths[0].read_bytes() == b"month-1"
        assert paths[1].read_bytes() == b"month-2"

    def test_download_year_skips_existing_files(self, http_file_server, tmp_path, monkeypatch):
        _, port = http_file_server

        def fake_build_month_url(year: int, month: int) -> str:
            return f"http://127.0.0.1:{port}/yellow_tripdata_{year}-{month:02d}.parquet"

        monkeypatch.setattr(async_io, "build_month_url", fake_build_month_url)

        output_dir = tmp_path / "downloads_skip"
        output_dir.mkdir()
        existing = output_dir / "yellow_tripdata_2023-01.parquet"
        pd.DataFrame({"value": [1]}).to_parquet(existing, index=False)

        paths = asyncio.run(async_io.download_year(2023, output_dir, concurrency=2, months=[1, 2]))

        assert len(paths) == 2
        assert pd.read_parquet(existing).to_dict(orient="list") == {"value": [1]}
        assert paths[1].read_bytes() == b"month-2"

    def test_invalid_existing_file_is_redownloaded(self, http_file_server, tmp_path, monkeypatch):
        _, port = http_file_server

        def fake_build_month_url(year: int, month: int) -> str:
            return f"http://127.0.0.1:{port}/yellow_tripdata_{year}-{month:02d}.parquet"

        monkeypatch.setattr(async_io, "build_month_url", fake_build_month_url)

        output_dir = tmp_path / "downloads_redownload"
        output_dir.mkdir()
        broken = output_dir / "yellow_tripdata_2023-01.parquet"
        broken.write_bytes(b"broken")

        paths = asyncio.run(async_io.download_year(2023, output_dir, concurrency=1, months=[1]))

        assert len(paths) == 1
        assert paths[0].read_bytes() == b"month-1"
