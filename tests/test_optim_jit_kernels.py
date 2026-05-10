"""
tests/test_optim_jit_kernels.py
-------------------------------
Tests pour les noyaux Numba.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optim.jit_kernels import (
    haversine_numba,
    haversine_numba_parallel,
    haversine_numpy,
    haversine_python,
    weighted_rolling_median_numba,
)


class TestHaversineImplementations:
    def test_all_implementations_match(self):
        lat1 = np.array([48.8566, 40.7128, 51.5074], dtype=np.float64)
        lon1 = np.array([2.3522, -74.0060, -0.1278], dtype=np.float64)
        lat2 = np.array([45.7640, 34.0522, 52.5200], dtype=np.float64)
        lon2 = np.array([4.8357, -118.2437, 13.4050], dtype=np.float64)

        python_result = haversine_python(lat1, lon1, lat2, lon2)
        numpy_result = haversine_numpy(lat1, lon1, lat2, lon2)
        numba_result = haversine_numba(lat1, lon1, lat2, lon2)
        numba_parallel_result = haversine_numba_parallel(lat1, lon1, lat2, lon2)

        assert np.allclose(python_result, numpy_result)
        assert np.allclose(python_result, numba_result)
        assert np.allclose(python_result, numba_parallel_result)


class TestWeightedRollingMedian:
    def test_weighted_rolling_median_numba(self):
        values = np.array([1.0, 5.0, 2.0, 8.0, 3.0], dtype=np.float64)
        weights = np.array([1.0, 1.0, 2.0], dtype=np.float64)

        result = weighted_rolling_median_numba(values, weights, window=3)

        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(5.0)
        assert result[4] == pytest.approx(3.0)
