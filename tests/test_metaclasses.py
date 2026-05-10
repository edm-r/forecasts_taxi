"""
tests/test_metaclasses.py
--------------------------
Tests unitaires pour PipelineStepMeta et BasePipelineStep.

Exercice 1.1 — Master 1 IABD

Lancement :
    pytest tests/test_metaclasses.py -v
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.metaclasses import BasePipelineStep, PipelineStepMeta


def make_valid_step(step_name: str = "valid_step"):
    """Fabrique une sous-classe valide à la volée."""

    class ValidStep(BasePipelineStep):
        name = step_name

        def _run(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    return ValidStep


class RecorderObserver:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def update(self, event: dict) -> None:
        self.events.append(event)


class TestValidStep:
    def test_definition_does_not_raise(self):
        step_cls = make_valid_step("loader")
        assert step_cls is not None

    def test_instance_can_be_created(self):
        step_cls = make_valid_step("cleaner")
        step = step_cls()
        assert isinstance(step, BasePipelineStep)

    def test_run_returns_dataframe(self):
        step = make_valid_step("transformer")()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = step.run(df)
        pd.testing.assert_frame_equal(result, df)

    def test_repr_contains_name(self):
        step = make_valid_step("my_step")()
        assert "my_step" in repr(step)

    def test_metaclass_is_pipelinestepmeta(self):
        step_cls = make_valid_step()
        assert type(step_cls) is PipelineStepMeta


class TestObserverLifecycle:
    def test_run_notifies_before_and_after(self):
        observer = RecorderObserver()
        step = make_valid_step("observed")()
        step.attach(observer)

        df = pd.DataFrame({"a": [1, 2]})
        result = step.run(df)

        pd.testing.assert_frame_equal(result, df)
        assert [event["event"] for event in observer.events] == ["before_run", "after_run"]
        assert observer.events[0]["step"] == "observed"
        assert observer.events[1]["rows_out"] == 2

    def test_detach_stops_notifications(self):
        observer = RecorderObserver()
        step = make_valid_step("silent")()
        step.attach(observer)
        step.detach(observer)

        step.run(pd.DataFrame({"a": [1]}))
        assert observer.events == []

    def test_invalid_observer_raises(self):
        step = make_valid_step("invalid_observer")()
        step.attach(object())

        with pytest.raises(TypeError, match="update"):
            step.run(pd.DataFrame({"a": [1]}))


class TestMissingName:
    def test_raises_type_error_at_class_definition(self):
        with pytest.raises(TypeError, match="name"):
            class StepWithoutName(BasePipelineStep):
                def _run(self, df: pd.DataFrame) -> pd.DataFrame:
                    return df

    def test_name_must_be_str(self):
        with pytest.raises(TypeError, match="str"):
            class IntNameStep(BasePipelineStep):
                name = 42

                def _run(self, df: pd.DataFrame) -> pd.DataFrame:
                    return df


class TestRunHookContract:
    def test_missing_run_hook_raises(self):
        with pytest.raises(TypeError, match="_run"):
            class StepWithoutHook(BasePipelineStep):
                name = "missing_hook"

    def test_run_override_is_rejected(self):
        with pytest.raises(TypeError, match=r"redéfinir 'run\(\)'"):
            class RunOverride(BasePipelineStep):
                name = "bad_run"

                def run(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
                    return df

    def test_hook_must_be_callable(self):
        with pytest.raises(TypeError, match="callable"):
            class HookNotCallable(BasePipelineStep):
                name = "hook_not_callable"
                _run = "nope"

    def test_extra_argument_is_rejected(self):
        with pytest.raises(TypeError, match="signature"):
            class TooManyArgs(BasePipelineStep):
                name = "too_many"

                def _run(self, df: pd.DataFrame, extra) -> pd.DataFrame:  # type: ignore[override]
                    return df

    def test_missing_df_annotation_is_rejected(self):
        with pytest.raises(TypeError, match="annoter 'df'"):
            class MissingDfAnnotation(BasePipelineStep):
                name = "missing_df_annotation"

                def _run(self, df) -> pd.DataFrame:  # type: ignore[override]
                    return df

    def test_wrong_df_annotation_is_rejected(self):
        with pytest.raises(TypeError, match="annoter 'df'"):
            class WrongDfAnnotation(BasePipelineStep):
                name = "wrong_df_annotation"

                def _run(self, df: dict) -> pd.DataFrame:  # type: ignore[override]
                    return pd.DataFrame(df)

    def test_missing_return_annotation_is_rejected(self):
        with pytest.raises(TypeError, match="annoter son retour"):
            class MissingReturnAnnotation(BasePipelineStep):
                name = "missing_return_annotation"

                def _run(self, df: pd.DataFrame):  # type: ignore[override]
                    return df

    def test_wrong_return_annotation_is_rejected(self):
        with pytest.raises(TypeError, match="annoter son retour"):
            class WrongReturnAnnotation(BasePipelineStep):
                name = "wrong_return_annotation"

                def _run(self, df: pd.DataFrame) -> dict:  # type: ignore[override]
                    return {}


class TestBasePipelineStep:
    def test_base_class_can_be_imported(self):
        assert BasePipelineStep is not None

    def test_base_hook_raises_not_implemented(self):
        class ConcreteStep(BasePipelineStep):
            name = "concrete"

            def _run(self, df: pd.DataFrame) -> pd.DataFrame:
                return super()._run(df)

        with pytest.raises(NotImplementedError):
            ConcreteStep().run(pd.DataFrame())

    def test_run_rejects_non_dataframe_outputs(self):
        class BadOutputStep(BasePipelineStep):
            name = "bad_output"

            def _run(self, df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
                return "not a dataframe"  # type: ignore[return-value]

        with pytest.raises(TypeError, match="pd.DataFrame"):
            BadOutputStep().run(pd.DataFrame())


class TestInheritedHook:
    def test_subclass_can_inherit_a_valid_hook(self):
        Parent = make_valid_step("parent")

        class Child(Parent):
            name = "child"

        result = Child().run(pd.DataFrame({"a": [1]}))
        assert result.shape == (1, 1)
