"""
tests/test_decorators.py
------------------------
Tests unitaires pour les décorateurs avancés.
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.decorators import log_calls, memoize_dataframe, retry, timeit


class TestTimeit:
    def test_works_without_parentheses(self, caplog):
        @timeit
        def add(a, b):
            return a + b

        with caplog.at_level(logging.INFO):
            assert add(2, 3) == 5

        assert any("[timeit] add took" in message for message in caplog.messages)

    def test_works_with_parentheses(self, caplog):
        @timeit(unit="us")
        def mul(a, b):
            return a * b

        with caplog.at_level(logging.INFO):
            assert mul(2, 4) == 8

        assert any("us" in message for message in caplog.messages)


class TestLogCalls:
    def test_logs_arguments(self, caplog):
        @log_calls()
        def greet(name, punctuation="!"):
            return f"Hello {name}{punctuation}"

        with caplog.at_level(logging.INFO):
            assert greet("NYC", punctuation="?") == "Hello NYC?"

        assert any("name='NYC'" in message for message in caplog.messages)
        assert any("punctuation='?'" in message for message in caplog.messages)

    def test_can_hide_arguments(self, caplog):
        @log_calls(log_args=False)
        def ping(value):
            return value

        with caplog.at_level(logging.INFO):
            assert ping(42) == 42

        assert any("[log_calls] ping()" in message for message in caplog.messages)


class TestRetry:
    def test_retries_then_succeeds(self, monkeypatch):
        calls = {"count": 0}
        delays: list[float] = []

        monkeypatch.setattr("core.decorators.time.sleep", delays.append)

        @retry(max_attempts=3, backoff=2.0, exceptions=(ValueError,))
        def flaky():
            calls["count"] += 1
            if calls["count"] < 3:
                raise ValueError("temporary")
            return "ok"

        assert flaky() == "ok"
        assert calls["count"] == 3
        assert delays == [1.0, 2.0]

    def test_raises_after_max_attempts(self, monkeypatch):
        monkeypatch.setattr("core.decorators.time.sleep", lambda _: None)

        @retry(max_attempts=2, backoff=2.0, exceptions=(ValueError,))
        def always_fails():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            always_fails()

    def test_does_not_swallow_other_exceptions(self):
        @retry(max_attempts=3, exceptions=(ValueError,))
        def wrong_error():
            raise TypeError("wrong type")

        with pytest.raises(TypeError, match="wrong type"):
            wrong_error()


class TestMemoizeDataFrame:
    def test_same_dataframe_content_hits_cache(self):
        calls = {"count": 0}

        @memoize_dataframe(maxsize=2)
        def compute(df: pd.DataFrame, scale: int = 1) -> pd.DataFrame:
            calls["count"] += 1
            result = df.copy()
            result["value"] = result["value"] * scale
            return result

        df1 = pd.DataFrame({"value": [1, 2, 3]})
        out1 = compute(df1, scale=2)
        out1.loc[0, "value"] = 999

        out2 = compute(df1.copy(), scale=2)

        assert calls["count"] == 1
        assert out2.loc[0, "value"] == 2

    def test_lru_eviction_occurs(self):
        calls = {"count": 0}

        @memoize_dataframe(maxsize=1)
        def compute(df: pd.DataFrame) -> pd.DataFrame:
            calls["count"] += 1
            return df.copy()

        df1 = pd.DataFrame({"value": [1]})
        df2 = pd.DataFrame({"value": [2]})

        compute(df1)
        compute(df2)
        compute(df1)

        assert calls["count"] == 3
