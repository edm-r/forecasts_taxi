"""
tests/test_optim_benchmarks.py
------------------------------
Tests pour les helpers de benchmark.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optim.benchmarks import compare_functions, measure_peak_memory, profile_with_cprofile, time_call


def increment(value: int) -> int:
    return value + 1


def allocate_list(size: int) -> list[int]:
    return list(range(size))


class TestTimeCall:
    def test_time_call_returns_stats(self):
        result = time_call(increment, 1, repeats=2, warmup=0)
        assert result["function"] == "increment"
        assert len(result["samples_ms"]) == 2
        assert result["result"] == 2


class TestProfileWithCProfile:
    def test_profile_output_mentions_function(self):
        report = profile_with_cprofile(increment, 1, top_n=5)
        assert "increment" in report


class TestMeasurePeakMemory:
    def test_measure_peak_memory_returns_peak(self):
        report = measure_peak_memory(allocate_list, 1000)
        assert report["function"] == "allocate_list"
        assert report["peak_mb"] >= 0.0
        assert len(report["result"]) == 1000


class TestCompareFunctions:
    def test_compare_functions_returns_one_report_per_function(self):
        reports = compare_functions([increment, increment], 1, repeats=1, warmup=0)
        assert len(reports) == 2
        assert all(report["result"] == 2 for report in reports)
