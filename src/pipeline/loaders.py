"""
src/pipeline/loaders.py
-----------------------
Loaders de données et Factory basée sur un registre de classes.

Exercice 1.4 — Master 1 IABD

Architecture :
    @register_loader(".parquet")   ← décorateur qui peuple _LOADER_REGISTRY
    class ParquetLoader(BasePipelineStep):
        ...

    loader = LoaderFactory.create("data/raw/file.parquet")
    df     = loader.run(df=None)   # df ignoré : c'est le loader qui le crée
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from core.metaclasses import BasePipelineStep

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Registre global des loaders
# ══════════════════════════════════════════════════════════════════════════════

_LOADER_REGISTRY: dict[str, type[BasePipelineStep]] = {}


def register_loader(extension: str):
    """
    Décorateur de classe qui enregistre un loader pour une extension donnée.

    Usage ::

        @register_loader(".parquet")
        class ParquetLoader(BasePipelineStep):
            ...

    Cela évite toute cascade if/elif dans la factory.
    """
    def decorator(cls: type[BasePipelineStep]) -> type[BasePipelineStep]:
        if not issubclass(cls, BasePipelineStep):
            raise TypeError(
                f"Le loader '{cls.__name__}' doit hériter de BasePipelineStep."
            )
        ext = extension.lower()
        if ext in _LOADER_REGISTRY:
            logger.warning(
                "Extension %r déjà enregistrée par %s ; écrasement par %s.",
                ext,
                _LOADER_REGISTRY[ext].__name__,
                cls.__name__,
            )
        _LOADER_REGISTRY[ext] = cls
        return cls
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
#  Loaders concrets
# ══════════════════════════════════════════════════════════════════════════════

@register_loader(".parquet")
class ParquetLoader(BasePipelineStep):
    """
    Charge un fichier Parquet via PyArrow.

    Attributs configurables :
        columns  : liste de colonnes à charger (None = toutes)
        filters  : filtres row-group PyArrow (ex. [("year","==",2023)])
    """

    name = "parquet_loader"

    def __init__(
        self,
        path: str | Path,
        columns: list[str] | None = None,
        filters=None,
    ) -> None:
        self.path = Path(path)
        self.columns = columns
        self.filters = filters

    def _run(self, df: pd.DataFrame) -> pd.DataFrame:
        """``df`` est ignoré : le loader est la source de données."""
        logger.info("ParquetLoader : chargement de %s", self.path)
        result = pd.read_parquet(
            self.path,
            engine="pyarrow",
            columns=self.columns,
            filters=self.filters,
        )
        logger.info(
            "ParquetLoader : %d lignes × %d colonnes chargées.",
            len(result),
            len(result.columns),
        )
        return result


@register_loader(".csv")
class CSVLoader(BasePipelineStep):
    """
    Charge un fichier CSV avec ``pd.read_csv``.

    Attributs configurables :
        sep      : séparateur (défaut ",")
        encoding : encodage (défaut "utf-8")
        kwargs   : tout argument supplémentaire passé à read_csv
    """

    name = "csv_loader"

    def __init__(
        self,
        path: str | Path,
        sep: str = ",",
        encoding: str = "utf-8",
        **kwargs,
    ) -> None:
        self.path = Path(path)
        self.sep = sep
        self.encoding = encoding
        self._kwargs = kwargs

    def _run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("CSVLoader : chargement de %s", self.path)
        result = pd.read_csv(
            self.path,
            sep=self.sep,
            encoding=self.encoding,
            **self._kwargs,
        )
        logger.info(
            "CSVLoader : %d lignes × %d colonnes chargées.",
            len(result),
            len(result.columns),
        )
        return result


@register_loader(".json")
class JSONLoader(BasePipelineStep):
    """
    Charge un fichier JSON (orienté records ou autres orientations Pandas).

    Attributs configurables :
        orient   : orientation du JSON (None = auto-détection Pandas)
        lines    : True si JSON-Lines (un objet par ligne)
        kwargs   : tout argument supplémentaire passé à read_json
    """

    name = "json_loader"

    def __init__(
        self,
        path: str | Path,
        orient: str | None = None,
        lines: bool = False,
        **kwargs,
    ) -> None:
        self.path = Path(path)
        self.orient = orient
        self.lines = lines
        self._kwargs = kwargs

    def _run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("JSONLoader : chargement de %s", self.path)
        read_kwargs = {k: v for k, v in {
            "orient": self.orient,
            "lines": self.lines,
            **self._kwargs,
        }.items() if v is not None}
        result = pd.read_json(self.path, **read_kwargs)
        logger.info(
            "JSONLoader : %d lignes × %d colonnes chargées.",
            len(result),
            len(result.columns),
        )
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

class LoaderFactory:
    """
    Instancie dynamiquement le bon loader selon l'extension du fichier.

    Principe :
        - Aucune cascade if/elif.
        - Consultation du registre ``_LOADER_REGISTRY`` peuplé par
          ``@register_loader``.
        - Extensible : un tiers peut enregistrer son propre loader sans
          modifier ce fichier.

    Exemple ::

        loader = LoaderFactory.create("data/raw/yellow_tripdata_2023-01.parquet")
        df = loader.run(df=None)

    Enregistrement d'un loader custom ::

        @register_loader(".feather")
        class FeatherLoader(BasePipelineStep):
            name = "feather_loader"
            def __init__(self, path, **kw): ...
            def _run(self, df): return pd.read_feather(self.path)
    """

    @staticmethod
    def create(path: str | Path, **loader_kwargs) -> BasePipelineStep:
        """
        Crée et renvoie le loader approprié pour ``path``.

        Parameters
        ----------
        path : str | Path
            Chemin vers le fichier de données.
        **loader_kwargs :
            Arguments supplémentaires transmis au constructeur du loader
            (ex. ``columns=[...]``, ``sep=";"``).

        Raises
        ------
        ValueError
            Si l'extension n'est pas enregistrée dans le registre.
        FileNotFoundError
            Si le fichier n'existe pas sur le disque.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        ext = path.suffix.lower()
        loader_cls = _LOADER_REGISTRY.get(ext)

        if loader_cls is None:
            supported = ", ".join(sorted(_LOADER_REGISTRY.keys()))
            raise ValueError(
                f"Aucun loader enregistré pour l'extension {ext!r}. "
                f"Extensions supportées : {supported}."
            )

        return loader_cls(path, **loader_kwargs)

    @staticmethod
    def supported_extensions() -> list[str]:
        """Renvoie la liste des extensions actuellement supportées."""
        return sorted(_LOADER_REGISTRY.keys())

    @staticmethod
    def registry() -> dict[str, str]:
        """Renvoie un snapshot lisible du registre {extension: classe}."""
        return {ext: cls.__name__ for ext, cls in _LOADER_REGISTRY.items()}
