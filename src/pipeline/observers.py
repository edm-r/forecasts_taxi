"""
src/pipeline/observers.py
-------------------------
Observer pattern pour le monitoring du pipeline NYC Taxi.

Exercice 1.6 — Master 1 IABD
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any


class PipelineSubject:
    """
    Sujet observable minimaliste.

    Les étapes du pipeline héritent de ce sujet pour notifier des composants
    tiers sans couplage fort avec la logique métier.
    """

    def _ensure_observers(self) -> list[Any]:
        if not hasattr(self, "_observers"):
            self._observers: list[Any] = []
        return self._observers

    def attach(self, observer: Any) -> None:
        observers = self._ensure_observers()
        if observer not in observers:
            observers.append(observer)

    def detach(self, observer: Any) -> None:
        observers = self._ensure_observers()
        if observer in observers:
            observers.remove(observer)

    def notify(self, event: dict[str, Any]) -> None:
        for observer in list(self._ensure_observers()):
            update = getattr(observer, "update", None)
            if not callable(update):
                raise TypeError(
                    f"L'observateur {observer!r} doit exposer une méthode update(event)."
                )
            update(event)


class LoggingObserver:
    """
    Observateur qui persiste les événements dans un fichier de log.
    """

    def __init__(self, log_path: str | Path = "pipeline.log") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        logger_name = f"pipeline.logging.{self.log_path.resolve()}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename) == self.log_path.resolve()
            for handler in self.logger.handlers
        ):
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self.logger.addHandler(handler)

    def update(self, event: dict[str, Any]) -> None:
        event_type = event.get("event", "unknown")
        step = event.get("step", "?")

        if event_type == "before_run":
            self.logger.info(
                "[%s] start rows_in=%s",
                step,
                event.get("rows_in"),
            )
            return

        if event_type == "after_run":
            self.logger.info(
                "[%s] done rows_in=%s rows_out=%s duration_ms=%.3f",
                step,
                event.get("rows_in"),
                event.get("rows_out"),
                event.get("duration_ms", 0.0),
            )
            return

        if event_type == "run_error":
            self.logger.error(
                "[%s] error duration_ms=%.3f error=%s",
                step,
                event.get("duration_ms", 0.0),
                event.get("error"),
            )
            return

        self.logger.info("[%s] event=%s payload=%s", step, event_type, event)


class MetricsObserver:
    """
    Observateur qui collecte des métriques simples par étape.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def update(self, event: dict[str, Any]) -> None:
        if event.get("event") != "after_run":
            return

        self.records.append(
            {
                "step": event.get("step"),
                "step_class": event.get("step_class"),
                "rows_in": event.get("rows_in"),
                "rows_out": event.get("rows_out"),
                "duration_ms": event.get("duration_ms"),
            }
        )


class AlertObserver:
    """
    Observateur qui alerte quand une étape supprime trop de lignes.
    """

    def __init__(self, drop_threshold: float = 0.30) -> None:
        self.drop_threshold = drop_threshold
        self.alerts: list[str] = []

    def update(self, event: dict[str, Any]) -> None:
        if event.get("event") != "after_run":
            return

        rows_in = event.get("rows_in")
        rows_out = event.get("rows_out")
        if not rows_in or rows_out is None:
            return

        drop_ratio = 1.0 - (rows_out / rows_in)
        if drop_ratio <= self.drop_threshold:
            return

        message = (
            f"L'étape '{event.get('step')}' a perdu {drop_ratio:.1%} des lignes "
            f"({rows_in} -> {rows_out})."
        )
        self.alerts.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
