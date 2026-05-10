"""
tests/test_orchestrator.py
--------------------------
Tests unitaires pour les mixins et l'orchestrateur Pipeline.
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.metaclasses import BasePipelineStep
from pipeline.observers import MetricsObserver
from pipeline.orchestrator import Pipeline


class LoaderStep(BasePipelineStep):
    name = "loader_step"

    def _run(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"fare_amount": [10.0, 20.0], "tip_amount": [1.0, 3.0]})


class TipPctStep(BasePipelineStep):
    name = "tip_pct_step"

    def _run(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["tip_pct"] = result["tip_amount"] / result["fare_amount"]
        return result


class DropLastRowStep(BasePipelineStep):
    name = "drop_last_row"

    def _run(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[:-1].copy()


class TestPipeline:
    def test_runs_steps_in_sequence(self):
        pipeline = Pipeline([LoaderStep(), TipPctStep(), DropLastRowStep()])

        result = pipeline.run(None)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["fare_amount", "tip_amount", "tip_pct"]
        assert len(result) == 1

    def test_attach_observer_propagates_to_steps(self):
        pipeline = Pipeline([LoaderStep(), TipPctStep()])
        metrics = MetricsObserver()
        pipeline.attach_observer(metrics)

        pipeline.run(None)

        assert len(metrics.records) == 2
        assert metrics.records[0]["step"] == "loader_step"
        assert metrics.records[1]["step"] == "tip_pct_step"

    def test_add_step_rejects_invalid_objects(self):
        pipeline = Pipeline()
        with pytest.raises(TypeError, match="BasePipelineStep"):
            pipeline.add_step(object())  # type: ignore[arg-type]

    def test_validate_rejects_invalid_input_type(self):
        pipeline = Pipeline([LoaderStep()])
        with pytest.raises(TypeError, match="pd.DataFrame ou None"):
            pipeline.validate("not a dataframe")  # type: ignore[arg-type]

    def test_serialization_roundtrip(self, tmp_path):
        pipeline = Pipeline([LoaderStep(), TipPctStep()])
        path = pipeline.to_pickle(tmp_path / "pipeline.pkl")

        restored = Pipeline.from_pickle(path)

        assert isinstance(restored, Pipeline)
        assert [step.name for step in restored.steps] == ["loader_step", "tip_pct_step"]

    def test_logs_pipeline_steps(self, caplog):
        pipeline = Pipeline([LoaderStep()])

        with caplog.at_level(logging.INFO):
            pipeline.run(None)

        assert any("Pipeline step=loader_step" in message for message in caplog.messages)
