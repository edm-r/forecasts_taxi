"""
tests/test_observers.py
-----------------------
Tests unitaires pour l'Observer du pipeline.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.metaclasses import BasePipelineStep
from pipeline.observers import (
    AlertObserver,
    LoggingObserver,
    MetricsObserver,
    PipelineSubject,
)


class IdentityStep(BasePipelineStep):
    name = "identity"

    def _run(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()


class DropRowsStep(BasePipelineStep):
    name = "drop_rows"

    def _run(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[:1].copy()


class RecorderObserver:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def update(self, event: dict) -> None:
        self.events.append(event)


class TestPipelineSubject:
    def test_attach_notify_detach(self):
        subject = PipelineSubject()
        observer = RecorderObserver()

        subject.attach(observer)
        subject.notify({"event": "ping"})
        subject.detach(observer)
        subject.notify({"event": "pong"})

        assert observer.events == [{"event": "ping"}]

    def test_duplicate_attach_is_ignored(self):
        subject = PipelineSubject()
        observer = RecorderObserver()

        subject.attach(observer)
        subject.attach(observer)
        subject.notify({"event": "once"})

        assert observer.events == [{"event": "once"}]


class TestLoggingObserver:
    def test_writes_log_file(self, tmp_path):
        log_path = tmp_path / "pipeline.log"
        observer = LoggingObserver(log_path)
        step = IdentityStep()
        step.attach(observer)

        step.run(pd.DataFrame({"a": [1, 2]}))

        content = log_path.read_text(encoding="utf-8")
        assert "[identity] start" in content
        assert "[identity] done" in content


class TestMetricsObserver:
    def test_collects_after_run_metrics(self):
        observer = MetricsObserver()
        step = IdentityStep()
        step.attach(observer)

        result = step.run(pd.DataFrame({"a": [1, 2, 3]}))

        assert len(observer.records) == 1
        assert observer.records[0]["step"] == "identity"
        assert observer.records[0]["rows_out"] == len(result)


class TestAlertObserver:
    def test_warns_on_large_row_drop(self):
        observer = AlertObserver(drop_threshold=0.30)
        step = DropRowsStep()
        step.attach(observer)

        with pytest.warns(RuntimeWarning, match="perdu"):
            step.run(pd.DataFrame({"a": [1, 2, 3, 4]}))

        assert len(observer.alerts) == 1

    def test_no_warning_on_small_drop(self):
        observer = AlertObserver(drop_threshold=0.90)
        step = DropRowsStep()
        step.attach(observer)

        step.run(pd.DataFrame({"a": [1, 2, 3, 4]}))
        assert observer.alerts == []
