"""
src/core/metaclasses.py
-----------------------
Metaclasse de validation et classe de base des étapes du pipeline NYC Taxi.

Exercice 1.1 — Master 1 IABD
"""

from __future__ import annotations

import inspect
from time import perf_counter
from typing import Any, get_type_hints

import pandas as pd

from pipeline.observers import PipelineSubject


class PipelineStepMeta(type):
    """
    Valide, à la définition de la classe, le contrat des étapes du pipeline.

    Contrat retenu :
    - chaque sous-classe concrète définit ``name: str`` ;
    - le point d'extension métier est ``_run(self, df: pd.DataFrame) -> pd.DataFrame`` ;
    - la méthode publique ``run()`` est fournie par ``BasePipelineStep`` pour
      centraliser l'instrumentation et la notification d'observateurs.
    """

    _EXPECTED_PARAMS = ("self", "df")
    _EXPECTED_HOOK = "_run"

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        if namespace.get("_is_base", False):
            return cls

        if "run" in namespace:
            raise TypeError(
                f"[PipelineStepMeta] '{name}' ne doit pas redéfinir 'run()'. "
                f"Implémentez '{mcs._EXPECTED_HOOK}(self, df: pd.DataFrame) -> pd.DataFrame' "
                "et laissez BasePipelineStep.run() gérer le cycle d'exécution."
            )

        class_name_attr = namespace.get("name")
        if class_name_attr is None:
            for base in bases:
                inherited_name = getattr(base, "name", None)
                if inherited_name is not None and not getattr(base, "_is_base", False):
                    class_name_attr = inherited_name
                    break

        if class_name_attr is None:
            raise TypeError(
                f"[PipelineStepMeta] La classe '{name}' doit définir un attribut "
                "de classe 'name: str'."
            )

        if not isinstance(class_name_attr, str):
            raise TypeError(
                f"[PipelineStepMeta] L'attribut 'name' de '{name}' doit être de type "
                f"'str', pas {type(class_name_attr).__name__!r}."
            )

        hook = namespace.get(mcs._EXPECTED_HOOK)
        if hook is None:
            hook = mcs._find_inherited_hook(bases)

        if hook is None:
            raise TypeError(
                f"[PipelineStepMeta] La classe '{name}' doit implémenter la méthode "
                f"'{mcs._EXPECTED_HOOK}(self, df: pd.DataFrame) -> pd.DataFrame'."
            )

        if not callable(hook):
            raise TypeError(
                f"[PipelineStepMeta] '{name}.{mcs._EXPECTED_HOOK}' doit être callable, "
                f"pas {type(hook).__name__!r}."
            )

        mcs._validate_signature(name, hook)
        mcs._validate_annotations(name, hook)
        return cls

    @classmethod
    def _find_inherited_hook(cls, bases: tuple[type, ...]) -> Any | None:
        base_step = globals().get("BasePipelineStep")
        abstract_hook = getattr(base_step, cls._EXPECTED_HOOK, None) if base_step else None

        for base in bases:
            inherited_hook = getattr(base, cls._EXPECTED_HOOK, None)
            if inherited_hook is not None and inherited_hook is not abstract_hook:
                return inherited_hook
        return None

    @classmethod
    def _validate_signature(cls, class_name: str, hook: Any) -> None:
        sig = inspect.signature(hook)
        params = tuple(sig.parameters.keys())
        if params != cls._EXPECTED_PARAMS:
            raise TypeError(
                f"[PipelineStepMeta] La méthode '{class_name}.{cls._EXPECTED_HOOK}' doit avoir "
                f"la signature {cls._EXPECTED_HOOK}(self, df), pas "
                f"{cls._EXPECTED_HOOK}({', '.join(params)})."
            )

    @classmethod
    def _validate_annotations(cls, class_name: str, hook: Any) -> None:
        try:
            hints = get_type_hints(hook, globalns=getattr(hook, "__globals__", {}), localns={"pd": pd})
        except Exception as exc:  # pragma: no cover
            raise TypeError(
                f"[PipelineStepMeta] Impossible de résoudre les annotations de "
                f"'{class_name}.{cls._EXPECTED_HOOK}': {exc}."
            ) from exc

        df_annotation = hints.get("df")
        return_annotation = hints.get("return")

        if df_annotation is not pd.DataFrame:
            raise TypeError(
                f"[PipelineStepMeta] '{class_name}.{cls._EXPECTED_HOOK}' doit annoter "
                "'df' avec 'pd.DataFrame'."
            )

        if return_annotation is not pd.DataFrame:
            raise TypeError(
                f"[PipelineStepMeta] '{class_name}.{cls._EXPECTED_HOOK}' doit annoter "
                "son retour avec 'pd.DataFrame'."
            )


class BasePipelineStep(PipelineSubject, metaclass=PipelineStepMeta):
    """
    Classe mère des étapes du pipeline.

    ``run()`` est la méthode publique stable. Les sous-classes implémentent
    uniquement ``_run()`` pour le travail métier.
    """

    _is_base = True

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        input_rows = len(df) if isinstance(df, pd.DataFrame) else None
        started_at = perf_counter()
        self.notify(
            {
                "event": "before_run",
                "step": self.name,
                "step_class": self.__class__.__name__,
                "rows_in": input_rows,
            }
        )

        try:
            result = self._run(df)
        except Exception as exc:
            duration_ms = (perf_counter() - started_at) * 1000
            self.notify(
                {
                    "event": "run_error",
                    "step": self.name,
                    "step_class": self.__class__.__name__,
                    "rows_in": input_rows,
                    "duration_ms": duration_ms,
                    "error": repr(exc),
                }
            )
            raise

        if not isinstance(result, pd.DataFrame):
            raise TypeError(
                f"La méthode '{self.__class__.__name__}._run' doit renvoyer un "
                f"pd.DataFrame, pas {type(result).__name__!r}."
            )

        duration_ms = (perf_counter() - started_at) * 1000
        self.notify(
            {
                "event": "after_run",
                "step": self.name,
                "step_class": self.__class__.__name__,
                "rows_in": input_rows,
                "rows_out": len(result),
                "duration_ms": duration_ms,
            }
        )
        return result

    def _run(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError(
            f"La classe '{self.__class__.__name__}' doit implémenter _run()."
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={getattr(self, 'name', '?')!r}>"
