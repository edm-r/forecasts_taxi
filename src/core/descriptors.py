"""
src/core/descriptors.py
-----------------------
Descripteurs de validation pour les classes de configuration du pipeline.

Exercice 1.2 — Master 1 IABD

Chaque descripteur implémente le protocole Python :
    __set_name__(owner, name)  → appelé lors de la définition de la classe
    __get__(obj, objtype)      → lecture de l'attribut
    __set__(obj, value)        → écriture avec validation
"""

from __future__ import annotations
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
#  Classe de base commune
# ══════════════════════════════════════════════════════════════════════════════

class _BaseDescriptor:
    """
    Socle commun à tous les descripteurs.

    - Stocke la valeur dans ``obj.__dict__`` avec une clé privée
      (``_<owner>__<attr>``) pour éviter les collisions de noms.
    - Délègue la validation à ``_validate``, à surcharger dans les sous-classes.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        # Clé de stockage privée : évite les récursions infinies et les
        # collisions entre descripteurs de même nom dans des classes différentes.
        self._public_name = name
        self._private_name = f"_{owner.__name__}__{name}"

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            # Accès depuis la classe elle-même → renvoie le descripteur
            return self
        return obj.__dict__.get(self._private_name)

    def __set__(self, obj: Any, value: Any) -> None:
        self._validate(value)
        obj.__dict__[self._private_name] = value

    def _validate(self, value: Any) -> None:  # pragma: no cover
        """À surcharger. Lève ValueError ou TypeError si invalide."""
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
#  Descripteurs concrets
# ══════════════════════════════════════════════════════════════════════════════

class Positive(_BaseDescriptor):
    """
    Accepte uniquement les nombres **strictement supérieurs** à ``min_value``.

    Exemple ::

        class Config:
            n_trees = Positive(min_value=10)

        cfg = Config()
        cfg.n_trees = 100   # OK
        cfg.n_trees = 5     # ValueError : 5 <= 10
        cfg.n_trees = -1    # ValueError
    """

    def __init__(self, min_value: float = 0) -> None:
        self.min_value = min_value

    def _validate(self, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"[Positive] '{self._public_name}' attend un nombre (int/float), "
                f"reçu {type(value).__name__!r}."
            )
        if value <= self.min_value:
            raise ValueError(
                f"[Positive] '{self._public_name}' doit être > {self.min_value}, "
                f"reçu {value}."
            )


class OneOf(_BaseDescriptor):
    """
    Accepte uniquement les valeurs présentes dans ``choices``.

    Exemple ::

        class Config:
            objective = OneOf("regression", "binary", "multiclass")

        cfg = Config()
        cfg.objective = "binary"      # OK
        cfg.objective = "clustering"  # ValueError
    """

    def __init__(self, *choices: Any) -> None:
        if not choices:
            raise ValueError("[OneOf] Au moins une valeur autorisée est requise.")
        self.choices = choices

    def _validate(self, value: Any) -> None:
        if value not in self.choices:
            raise ValueError(
                f"[OneOf] '{self._public_name}' doit être parmi "
                f"{self.choices}, reçu {value!r}."
            )


class TypedAttr(_BaseDescriptor):
    """
    Vérifie le **type strict** (``type(value) is expected_type``).

    Exemple ::

        class Config:
            max_depth = TypedAttr(int)

        cfg = Config()
        cfg.max_depth = 6       # OK
        cfg.max_depth = 6.0     # TypeError (float != int)
        cfg.max_depth = "six"   # TypeError
    """

    def __init__(self, expected_type: type) -> None:
        self.expected_type = expected_type

    def _validate(self, value: Any) -> None:
        if type(value) is not self.expected_type:
            raise TypeError(
                f"[TypedAttr] '{self._public_name}' doit être de type "
                f"{self.expected_type.__name__!r}, "
                f"reçu {type(value).__name__!r} ({value!r})."
            )


class BoundedFloat(_BaseDescriptor):
    """
    Accepte uniquement les **float** dans l'intervalle fermé **[low, high]**.

    Exemple ::

        class Config:
            learning_rate = BoundedFloat(0.0, 1.0)

        cfg = Config()
        cfg.learning_rate = 0.01   # OK
        cfg.learning_rate = 1.5    # ValueError
        cfg.learning_rate = "lr"   # TypeError
    """

    def __init__(self, low: float, high: float) -> None:
        if low > high:
            raise ValueError(
                f"[BoundedFloat] low ({low}) ne peut pas être > high ({high})."
            )
        self.low = low
        self.high = high

    def _validate(self, value: Any) -> None:
        if not isinstance(value, float):
            raise TypeError(
                f"[BoundedFloat] '{self._public_name}' attend un float, "
                f"reçu {type(value).__name__!r}."
            )
        if not (self.low <= value <= self.high):
            raise ValueError(
                f"[BoundedFloat] '{self._public_name}' doit être dans "
                f"[{self.low}, {self.high}], reçu {value}."
            )


# ══════════════════════════════════════════════════════════════════════════════
#  Démonstration : ModelConfig
# ══════════════════════════════════════════════════════════════════════════════

class ModelConfig:
    """
    Configuration d'un modèle ML avec validation automatique des attributs.

    Toute affectation invalide est rejetée *à l'exécution* par les descripteurs.

    Exemple ::

        cfg = ModelConfig()
        cfg.learning_rate = 0.05         # OK
        cfg.n_estimators  = 200          # OK
        cfg.objective     = "regression" # OK
        cfg.max_depth     = 8            # OK

        cfg.learning_rate = 2.0          # ValueError : hors [0.0, 1.0]
        cfg.n_estimators  = 5            # ValueError : <= 10
        cfg.objective     = "kmeans"     # ValueError : choix invalide
        cfg.max_depth     = 4.5          # TypeError  : float != int
    """

    learning_rate = BoundedFloat(0.0, 1.0)
    n_estimators  = Positive(min_value=10)
    objective     = OneOf("regression", "binary", "multiclass")
    max_depth     = TypedAttr(int)

    def __repr__(self) -> str:
        return (
            f"ModelConfig("
            f"learning_rate={self.learning_rate}, "
            f"n_estimators={self.n_estimators}, "
            f"objective={self.objective!r}, "
            f"max_depth={self.max_depth})"
        )
