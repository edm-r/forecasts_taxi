"""
tests/test_optim_parallel.py
----------------------------
Tests pour le parallélisme CPU et I/O.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optim.parallel import (
    benchmark_io_readers,
    process_months_parallel,
    process_months_sequential,
    read_parquet_files_sequential,
    read_parquet_files_threaded,
    run_cpu_bound_with_processes,
    run_cpu_bound_with_threads,
)


@pytest.fixture
def monthly_raw_data(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    for month in (1, 2):
        df = pd.DataFrame(
            {
                "tpep_pickup_datetime": pd.to_datetime(
                    ["2023-01-01 08:00:00", "2023-01-01 09:00:00"]
                ),
                "tpep_dropoff_datetime": pd.to_datetime(
                    ["2023-01-01 08:15:00", "2023-01-01 09:30:00"]
                ),
                "trip_distance": [1.5 * month, 2.0 * month],
                "fare_amount": [10.0, 20.0],
            }
        )
        df.to_parquet(raw_dir / f"yellow_tripdata_2023-{month:02d}.parquet", index=False)

    return raw_dir


class TestMonthProcessing:
    def test_process_months_sequential(self, monthly_raw_data, tmp_path):
        output_paths = process_months_sequential(
            [1, 2],
            raw_dir=monthly_raw_data,
            processed_dir=tmp_path / "processed",
        )

        assert len(output_paths) == 2
        for path in output_paths:
            assert path.exists()
            df = pd.read_parquet(path)
            assert "duration_minutes" in df.columns
            assert "avg_speed_mph" in df.columns
            assert "penalty" in df.columns

    def test_process_months_parallel(self, monthly_raw_data, tmp_path):
        output_paths = process_months_parallel(
            [1, 2],
            raw_dir=monthly_raw_data,
            processed_dir=tmp_path / "processed_parallel",
            max_workers=2,
        )

        assert len(output_paths) == 2
        assert all(path.exists() for path in output_paths)


class TestCpuBoundExecutors:
    def test_threads_return_expected_total(self):
        result = run_cpu_bound_with_threads(task_count=2, work_size=5000, max_workers=2)
        assert result["total"] == pytest.approx(10_000.0, rel=1e-9)

    def test_processes_return_expected_total(self):
        result = run_cpu_bound_with_processes(task_count=2, work_size=5000, max_workers=2)
        assert result["total"] == pytest.approx(10_000.0, rel=1e-9)


class TestIoReaders:
    def test_read_parquet_files_sequential_and_threaded(self, monthly_raw_data):
        paths = sorted(monthly_raw_data.glob("*.parquet"))
        sequential = read_parquet_files_sequential(paths)
        threaded = read_parquet_files_threaded(paths, max_workers=2)

        assert len(sequential) == len(threaded) == 2
        assert [len(df) for df in sequential] == [len(df) for df in threaded]

    def test_benchmark_io_readers(self, monthly_raw_data):
        report = benchmark_io_readers(sorted(monthly_raw_data.glob("*.parquet")), max_workers=2)
        assert report["sequential_s"] >= 0.0
        assert report["threaded_s"] >= 0.0
