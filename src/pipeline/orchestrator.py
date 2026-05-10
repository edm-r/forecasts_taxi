"""
src/pipeline/orchestrator.py
----------------------------
Mixins et composition pour orchestrer le pipeline NYC Taxi.

Exercice 1.9 — Master 1 IABD
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from core.metaclasses import BasePipelineStep


class LoggableMixin:
    """
    Fournit des helpers de logging simples.
    """

    @property
    def logger(self) -> logging.Logger:
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return logging.getLogger(logger_name)

    def log_info(self, message: str, *args: Any) -> None:
        self.logger.info(message, *args)

    def log_warning(self, message: str, *args: Any) -> None:
        self.logger.warning(message, *args)


class SerializableMixin:
    """
    Sérialisation binaire via pickle.
    """

    def to_pickle(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def from_pickle(cls, path: str | Path) -> Any:
        path = Path(path)
        with path.open("rb") as fh:
            instance = pickle.load(fh)

        if not isinstance(instance, cls):
            raise TypeError(
                f"Le pickle contient un objet de type {type(instance).__name__!r}, "
                f"attendu {cls.__name__!r}."
            )
        return instance


class ValidatableMixin:
    """
    Contrat minimal de validation.
    """

    def validate(self, df: pd.DataFrame | None) -> bool:  # pragma: no cover
        raise NotImplementedError("La validation doit être surchargée.")


class Pipeline(LoggableMixin, SerializableMixin, ValidatableMixin):
    """
    Orchestrateur composé d'une liste d'étapes ``BasePipelineStep``.
    """

    def __init__(self, steps: Iterable[BasePipelineStep] | None = None) -> None:
        self.steps: list[BasePipelineStep] = []
        self._observers: list[Any] = []

        for step in steps or []:
            self.add_step(step)

    def add_step(self, step: BasePipelineStep) -> None:
        if not isinstance(step, BasePipelineStep):
            raise TypeError(
                f"Une étape doit hériter de BasePipelineStep, reçu {type(step).__name__!r}."
            )

        self.steps.append(step)
        for observer in self._observers:
            step.attach(observer)

    def attach_observer(self, observer: Any) -> None:
        if observer not in self._observers:
            self._observers.append(observer)
            for step in self.steps:
                step.attach(observer)

    def detach_observer(self, observer: Any) -> None:
        if observer in self._observers:
            self._observers.remove(observer)
            for step in self.steps:
                step.detach(observer)

    def validate(self, df: pd.DataFrame | None) -> bool:
        if df is not None and not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Le pipeline attend un pd.DataFrame ou None en entrée, reçu {type(df).__name__!r}."
            )

        for step in self.steps:
            if not isinstance(step, BasePipelineStep):
                raise TypeError(
                    f"Étape invalide dans le pipeline: {type(step).__name__!r}."
                )
        return True

    def run(self, df: pd.DataFrame | None = None) -> pd.DataFrame | None:
        self.validate(df)
        current = df

        for step in self.steps:
            self.log_info("Pipeline step=%s class=%s", step.name, step.__class__.__name__)
            current = step.run(current)

        if current is not None and not isinstance(current, pd.DataFrame):
            raise TypeError(
                f"Le pipeline doit produire un pd.DataFrame, pas {type(current).__name__!r}."
            )
        return current

    def __repr__(self) -> str:
        step_names = [step.name for step in self.steps]
        return f"Pipeline(steps={step_names!r})"
