"""
tests/test_context.py
---------------------
Tests unitaires pour les gestionnaires de contexte.
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.context import MemoryGuard, Timer, open_parquet_files, temp_dtypes


class TestTimer:
    def test_logs_elapsed_time(self, caplog):
        with caplog.at_level(logging.INFO):
            with Timer("test-block") as timer:
                sum(range(1000))

        assert timer.elapsed >= 0.0
        assert any("[Timer] test-block" in message for message in caplog.messages)


class TestMemoryGuard:
    def test_allows_small_allocations(self):
        with MemoryGuard(threshold_mb=10.0) as guard:
            payload = bytearray(1024)

        assert guard.peak_mb >= 0.0
        assert len(payload) == 1024

    def test_raises_on_peak_over_threshold(self):
        with pytest.raises(MemoryError, match="Pic mémoire dépassé"):
            with MemoryGuard(threshold_mb=0.0001):
                payload = bytearray(200_000)
                assert len(payload) == 200_000


class TestTempDtypes:
    def test_restores_original_dtypes(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        original_dtype = df["value"].dtype

        with temp_dtypes(df, {"value": "float64"}) as temp_df:
            assert str(temp_df["value"].dtype) == "float64"

        assert df["value"].dtype == original_dtype

    def test_missing_column_raises(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        with pytest.raises(KeyError, match="absente"):
            with temp_dtypes(df, {"missing": "float64"}):
                pass


class TestOpenParquetFiles:
    def test_opens_and_closes_handles(self, tmp_path):
        df = pd.DataFrame({"value": [1, 2]})
        (tmp_path / "a.parquet").write_bytes(df.to_parquet(index=False))
        (tmp_path / "b.parquet").write_bytes(df.to_parquet(index=False))

        with open_parquet_files(tmp_path) as handles:
            assert len(handles) == 2
            assert all(not handle.closed for handle in handles)

        assert all(handle.closed for handle in handles)

    def test_closes_handles_on_exception(self, tmp_path):
        df = pd.DataFrame({"value": [1, 2]})
        (tmp_path / "a.parquet").write_bytes(df.to_parquet(index=False))

        handles = []
        with pytest.raises(RuntimeError, match="boom"):
            with open_parquet_files(tmp_path) as opened:
                handles = opened
                raise RuntimeError("boom")

        assert all(handle.closed for handle in handles)
