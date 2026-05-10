"""
src/pipeline/strategies.py
--------------------------
Pattern Strategy pour le feature engineering du pipeline NYC Taxi.

Exercice 1.5 — Master 1 IABD

Hiérarchie :
    EncodingStrategy (ABC)
        ├── OneHotStrategy
        ├── FrequencyEncodingStrategy
        └── TargetEncodingStrategy  (version simplifiée ; détail → ex. 3.6)

La classe FeatureEngineer reçoit une stratégie par injection de dépendance
et délègue le travail d'encodage sans jamais connaître les détails internes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Interface abstraite
# ══════════════════════════════════════════════════════════════════════════════

class EncodingStrategy(ABC):
    """
    Interface que toute stratégie d'encodage doit respecter.

    Une stratégie :
    - reçoit un DataFrame et le nom d'une colonne catégorielle,
    - renvoie un **nouveau** DataFrame enrichi (immutabilité).

    Les sous-classes peuvent stocker un état appris (mapping, statistiques)
    pour distinguer fit (apprentissage) et transform (application).
    """

    @abstractmethod
    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Encode ``column`` dans ``df`` et renvoie le DataFrame transformé.

        Parameters
        ----------
        df     : DataFrame source (non modifié sur place)
        column : nom de la colonne catégorielle à encoder

        Returns
        -------
        pd.DataFrame avec la(les) colonne(s) encodée(s)
        """

    def fit(self, df: pd.DataFrame, column: str, target: str | None = None) -> "EncodingStrategy":
        """
        Apprentissage optionnel du mapping avant l'encodage.
        Peut être surchargé par les stratégies avec état (TargetEncoding…).

        Retourne ``self`` pour permettre le chaînage ``strategy.fit(df).encode(df)``.
        """
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ══════════════════════════════════════════════════════════════════════════════
#  Stratégie 1 — One-Hot Encoding
# ══════════════════════════════════════════════════════════════════════════════

class OneHotStrategy(EncodingStrategy):
    """
    Encodage One-Hot via ``pd.get_dummies``.

    Pour une colonne à haute cardinalité (265 zones), cela génère 265 colonnes
    binaires. Utile en baseline mais coûteux en mémoire ; préférer
    FrequencyEncoding ou TargetEncoding en production.

    Paramètres
    ----------
    drop_first : bool
        Supprimer la première modalité pour éviter la multicolinéarité
        (utile pour les modèles linéaires).
    prefix : str | None
        Préfixe des nouvelles colonnes (défaut : nom de la colonne d'origine).
    """

    def __init__(self, drop_first: bool = False, prefix: str | None = None) -> None:
        self.drop_first = drop_first
        self.prefix = prefix

    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column not in df.columns:
            raise KeyError(f"[OneHotStrategy] Colonne {column!r} absente du DataFrame.")

        prefix = self.prefix or column
        logger.debug("OneHotStrategy : encodage de %r (%d modalités).", column, df[column].nunique())

        dummies = pd.get_dummies(
            df[column],
            prefix=prefix,
            drop_first=self.drop_first,
            dtype=np.uint8,
        )
        return pd.concat([df.drop(columns=[column]), dummies], axis=1)

    def __repr__(self) -> str:
        return f"OneHotStrategy(drop_first={self.drop_first})"


# ══════════════════════════════════════════════════════════════════════════════
#  Stratégie 2 — Frequency Encoding
# ══════════════════════════════════════════════════════════════════════════════

class FrequencyEncodingStrategy(EncodingStrategy):
    """
    Remplace chaque modalité par sa **fréquence relative** dans le dataset.

    Avantages :
    - 1 seule colonne produite, quelle que soit la cardinalité.
    - Capture l'importance relative des zones (zones fréquentes vs rares).

    Paramètre
    ---------
    normalize : bool
        True (défaut) → proportion dans [0, 1].
        False → compte brut.
    """

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize
        self._mapping: dict[Any, float] = {}

    def fit(self, df: pd.DataFrame, column: str, target: str | None = None) -> "FrequencyEncodingStrategy":
        freq = df[column].value_counts(normalize=self.normalize)
        self._mapping = freq.to_dict()
        logger.debug(
            "FrequencyEncodingStrategy.fit : %d modalités apprises pour %r.",
            len(self._mapping),
            column,
        )
        return self

    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column not in df.columns:
            raise KeyError(f"[FrequencyEncodingStrategy] Colonne {column!r} absente.")

        # Auto-fit si le mapping est vide
        if not self._mapping:
            self.fit(df, column)

        new_col = f"{column}_freq"
        result = df.copy()
        result[new_col] = result[column].map(self._mapping).fillna(0.0)
        result = result.drop(columns=[column])

        logger.debug(
            "FrequencyEncodingStrategy.encode : colonne %r → %r.",
            column,
            new_col,
        )
        return result

    def __repr__(self) -> str:
        return f"FrequencyEncodingStrategy(normalize={self.normalize}, fitted={bool(self._mapping)})"


# ══════════════════════════════════════════════════════════════════════════════
#  Stratégie 3 — Target Encoding (version simplifiée)
# ══════════════════════════════════════════════════════════════════════════════

class TargetEncodingStrategy(EncodingStrategy):
    """
    Remplace chaque modalité par la **moyenne de la cible** conditionnelle
    à cette modalité, avec un **lissage bayésien** pour les modalités rares.

    Formule (lissage) :
        ŷ_c = (n_c · ȳ_c + m · ȳ_global) / (n_c + m)

    où :
        n_c      = nombre d'observations pour la modalité c
        ȳ_c      = moyenne de la cible pour c
        ȳ_global = moyenne globale de la cible
        m        = hyperparamètre de lissage (défaut = 10)

    Note : la version complète avec cross-fitting est à l'exercice 3.6.

    Paramètres
    ----------
    smoothing : float
        Paramètre m du lissage bayésien (plus grand → plus de régularisation).
    target_col : str | None
        Nom de la colonne cible. Doit être fourni lors du ``fit``.
    """

    def __init__(self, smoothing: float = 10.0, target_col: str | None = None) -> None:
        self.smoothing = smoothing
        self.target_col = target_col
        self._mapping: dict[Any, float] = {}
        self._global_mean: float = 0.0

    def fit(self, df: pd.DataFrame, column: str, target: str | None = None) -> "TargetEncodingStrategy":
        col_target = target or self.target_col
        if col_target is None:
            raise ValueError(
                "[TargetEncodingStrategy] Le nom de la colonne cible "
                "doit être fourni via fit(target=...) ou target_col=..."
            )
        if col_target not in df.columns:
            raise KeyError(f"[TargetEncodingStrategy] Cible {col_target!r} absente du DataFrame.")

        self._global_mean = df[col_target].mean()
        stats = df.groupby(column)[col_target].agg(["mean", "count"])

        # Lissage bayésien
        smoothed = (
            (stats["count"] * stats["mean"] + self.smoothing * self._global_mean)
            / (stats["count"] + self.smoothing)
        )
        self._mapping = smoothed.to_dict()
        logger.debug(
            "TargetEncodingStrategy.fit : %d modalités pour %r, global_mean=%.4f.",
            len(self._mapping),
            column,
            self._global_mean,
        )
        return self

    def encode(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        if column not in df.columns:
            raise KeyError(f"[TargetEncodingStrategy] Colonne {column!r} absente.")
        if not self._mapping:
            raise RuntimeError(
                "[TargetEncodingStrategy] fit() doit être appelé avant encode()."
            )

        new_col = f"{column}_target_enc"
        result = df.copy()
        # Modalités inconnues (test set) → on impute avec la moyenne globale
        result[new_col] = result[column].map(self._mapping).fillna(self._global_mean)
        result = result.drop(columns=[column])

        logger.debug(
            "TargetEncodingStrategy.encode : colonne %r → %r.",
            column,
            new_col,
        )
        return result

    def __repr__(self) -> str:
        return (
            f"TargetEncodingStrategy(smoothing={self.smoothing}, "
            f"target_col={self.target_col!r}, fitted={bool(self._mapping)})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  FeatureEngineer — injection de dépendance
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Transformateur de features qui délègue l'encodage à une ``EncodingStrategy``.

    La stratégie est injectée dans le constructeur → le code appelant ne
    connaît pas les détails de l'encodage, et on peut changer de stratégie
    **à chaud** via ``set_strategy``.

    Exemple ::

        fe = FeatureEngineer(strategy=OneHotStrategy())
        df_encoded = fe.transform(df, column="PULocationID")

        # Changement à chaud
        fe.set_strategy(FrequencyEncodingStrategy())
        df_encoded2 = fe.transform(df, column="PULocationID")
    """

    def __init__(self, strategy: EncodingStrategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> EncodingStrategy:
        return self._strategy

    def set_strategy(self, strategy: EncodingStrategy) -> None:
        """
        Remplace la stratégie courante.
        Peut être appelé entre deux transformations sans recréer l'objet.
        """
        logger.info(
            "FeatureEngineer : changement de stratégie %s → %s.",
            type(self._strategy).__name__,
            type(strategy).__name__,
        )
        self._strategy = strategy

    def fit(self, df: pd.DataFrame, column: str, target: str | None = None) -> "FeatureEngineer":
        """Entraîne la stratégie courante."""
        self._strategy.fit(df, column, target=target)
        return self

    def transform(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Applique l'encodage via la stratégie courante."""
        logger.info(
            "FeatureEngineer.transform : colonne=%r, stratégie=%s.",
            column,
            type(self._strategy).__name__,
        )
        return self._strategy.encode(df, column)

    def fit_transform(
        self,
        df: pd.DataFrame,
        column: str,
        target: str | None = None,
    ) -> pd.DataFrame:
        """Enchaîne fit puis transform."""
        return self.fit(df, column, target=target).transform(df, column)

    def __repr__(self) -> str:
        return f"FeatureEngineer(strategy={self._strategy!r})"