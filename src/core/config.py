"""
src/core/config.py
------------------
Deux implémentations du pattern Singleton pour la configuration globale
du pipeline NYC Taxi.

Exercice 1.3 — Master 1 IABD

  • Méthode A : surcharge de __new__  → ConfigNew
  • Méthode B : métaclasse SingletonMeta → Config  (version recommandée)

La classe ``Config`` (Méthode B) est celle utilisée dans tout le pipeline.
Elle charge un fichier ``config.yaml`` et expose ses valeurs en lecture seule.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

# PyYAML est requis par le projet ; le fallback sert uniquement à produire
# une erreur explicite si l'environnement est incomplet.
try:
    import yaml
    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False


# ──────────────────────────────────────────────────────────────────────────────
#  Valeurs par défaut (utilisées si config.yaml est absent)
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "paths": {
        "raw_data":       "data/raw",
        "processed_data": "data/processed",
        "features_data":  "data/features",
    },
    "model": {
        "learning_rate": 0.05,
        "n_estimators":  500,
        "max_depth":     6,
        "objective":     "regression",
        "seed":          42,
    },
    "pipeline": {
        "n_jobs": -1,
        "chunk_size": 100_000,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Méthode A — Surcharge de __new__
# ══════════════════════════════════════════════════════════════════════════════

class ConfigNew:
    """
    Singleton implémenté par surcharge de ``__new__``.

    Avantage  : simple, idiomatique Python.
    Limite    : ``__init__`` est rappelé à chaque ``ConfigNew()`` ; il faut
                un garde ``_initialized`` pour ne pas écraser l'état.
    Testabilité : appeler ``ConfigNew._instance = None`` pour réinitialiser.
    """

    _instance: "ConfigNew | None" = None
    _lock = threading.Lock()

    def __new__(cls, config_path: str | Path | None = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str | Path | None = None) -> None:
        if self._initialized:
            return                      # Ne ré-initialise pas si déjà fait
        self._data = _load_yaml(config_path)
        self._initialized = True

    # ── Propriétés en lecture seule ───────────────────────────────────────

    @property
    def raw_data_path(self) -> Path:
        return Path(self._data["paths"]["raw_data"])

    @raw_data_path.setter
    def raw_data_path(self, _):
        raise AttributeError("Config est en lecture seule.")

    @property
    def processed_data_path(self) -> Path:
        return Path(self._data["paths"]["processed_data"])

    @processed_data_path.setter
    def processed_data_path(self, _):
        raise AttributeError("Config est en lecture seule.")

    @property
    def features_data_path(self) -> Path:
        return Path(self._data["paths"]["features_data"])

    @features_data_path.setter
    def features_data_path(self, _):
        raise AttributeError("Config est en lecture seule.")

    @property
    def seed(self) -> int:
        return int(self._data["model"]["seed"])

    @seed.setter
    def seed(self, _):
        raise AttributeError("Config est en lecture seule.")

    @property
    def model(self) -> dict:
        return dict(self._data["model"])        # Copie défensive

    @model.setter
    def model(self, _):
        raise AttributeError("Config est en lecture seule.")

    @property
    def pipeline(self) -> dict:
        return dict(self._data["pipeline"])

    @pipeline.setter
    def pipeline(self, _):
        raise AttributeError("Config est en lecture seule.")

    def __repr__(self) -> str:
        return f"ConfigNew(seed={self.seed}, raw={self.raw_data_path})"


# ══════════════════════════════════════════════════════════════════════════════
#  Méthode B — Métaclasse SingletonMeta  (approche recommandée)
# ══════════════════════════════════════════════════════════════════════════════

class SingletonMeta(type):
    """
    Métaclasse Singleton thread-safe.

    Avantage par rapport à la Méthode A :
    - ``__init__`` n'est **jamais** rappelé sur l'instance existante.
    - Fonctionne pour n'importe quelle classe sans modifier son code.
    - Testabilité : ``SingletonMeta._instances.pop(MyClass, None)`` suffit.
    """

    _instances: dict[type, Any] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    """
    Configuration globale du pipeline (Singleton, Méthode B).

    Chargement ::

        config = Config()                          # 1er appel → charge YAML
        config = Config()                          # Appels suivants → même instance
        assert Config() is Config()                # True

    Réinitialisation pour les tests ::

        SingletonMeta._instances.pop(Config, None)

    Propriétés disponibles (toutes en lecture seule) :
        - raw_data_path          (Path)
        - processed_data_path    (Path)
        - features_data_path     (Path)
        - seed                   (int)
        - model                  (dict)
        - pipeline               (dict)
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._data = _load_yaml(config_path)

    # ── Propriétés en lecture seule ───────────────────────────────────────

    @property
    def raw_data_path(self) -> Path:
        return Path(self._data["paths"]["raw_data"])

    @raw_data_path.setter
    def raw_data_path(self, _):
        raise AttributeError("Config est en lecture seule.")

    @property
    def processed_data_path(self) -> Path:
        return Path(self._data["paths"]["processed_data"])

    @processed_data_path.setter
    def processed_data_path(self, _):
        raise AttributeError("Config est en lecture seule.")

    @property
    def features_data_path(self) -> Path:
        return Path(self._data["paths"]["features_data"])

    @features_data_path.setter
    def features_data_path(self, _):
        raise AttributeError("Config est en lecture seule.")

    @property
    def seed(self) -> int:
        return int(self._data["model"]["seed"])

    @seed.setter
    def seed(self, _):
        raise AttributeError("Config est en lecture seule.")

    @property
    def model(self) -> dict:
        return dict(self._data["model"])

    @model.setter
    def model(self, _):
        raise AttributeError("Config est en lecture seule.")

    @property
    def pipeline(self) -> dict:
        return dict(self._data["pipeline"])

    @pipeline.setter
    def pipeline(self, _):
        raise AttributeError("Config est en lecture seule.")

    def __repr__(self) -> str:
        return f"Config(seed={self.seed}, raw={self.raw_data_path})"


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers internes
# ══════════════════════════════════════════════════════════════════════════════

def _load_yaml(config_path: str | Path | None) -> dict:
    """
    Charge le fichier YAML fourni, ou cherche ``config.yaml`` à la racine
    du projet. Fusionne avec les valeurs par défaut (_DEFAULTS) si des clés
    sont manquantes.
    """
    import copy
    data = copy.deepcopy(_DEFAULTS)

    if config_path is None:
        # Chercher config.yaml à la racine du projet (deux niveaux au-dessus)
        root = Path(__file__).resolve().parent.parent.parent
        config_path = root / "config.yaml"

    config_path = Path(config_path)

    if config_path.exists():
        if not _HAS_YAML:
            raise RuntimeError(
                "PyYAML est requis pour charger config.yaml. "
                "Ajoutez 'PyYAML' aux dépendances du projet."
            )

        with open(config_path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        data = _deep_merge(data, loaded)

    return data


def _deep_merge(base: dict, override: dict) -> dict:
    """Fusion récursive de deux dicts : ``override`` écrase ``base``."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result
