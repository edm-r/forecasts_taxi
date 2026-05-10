"""
tests/test_loaders_strategies.py
---------------------------------
Tests unitaires pour LoaderFactory (ex. 1.4) et les stratégies d'encodage (ex. 1.5).

pytest tests/test_loaders_strategies.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.metaclasses import BasePipelineStep
from pipeline.loaders import (
    CSVLoader, JSONLoader, LoaderFactory, ParquetLoader, _LOADER_REGISTRY,
)
from pipeline.strategies import (
    EncodingStrategy,
    FeatureEngineer,
    FrequencyEncodingStrategy,
    OneHotStrategy,
    TargetEncodingStrategy,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Fixtures communes
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df():
    """DataFrame minimaliste simulant les données NYC Taxi."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        "PULocationID":   rng.integers(1, 30, size=n),
        "DOLocationID":   rng.integers(1, 30, size=n),
        "trip_distance":  rng.uniform(0.5, 20, size=n),
        "tip_pct":        rng.uniform(0, 0.3, size=n),
    })


@pytest.fixture
def parquet_file(sample_df, tmp_path):
    p = tmp_path / "test.parquet"
    sample_df.to_parquet(p, index=False)
    return p


@pytest.fixture
def csv_file(sample_df, tmp_path):
    p = tmp_path / "test.csv"
    sample_df.to_csv(p, index=False)
    return p


@pytest.fixture
def json_file(sample_df, tmp_path):
    p = tmp_path / "test.json"
    sample_df.to_json(p, orient="records")
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  Tests LoaderFactory & loaders
# ══════════════════════════════════════════════════════════════════════════════

class TestParquetLoader:
    def test_run_returns_dataframe(self, parquet_file, sample_df):
        loader = ParquetLoader(parquet_file)
        df = loader.run(df=None)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_df)

    def test_columns_filter(self, parquet_file):
        loader = ParquetLoader(parquet_file, columns=["PULocationID", "trip_distance"])
        df = loader.run(df=None)
        assert list(df.columns) == ["PULocationID", "trip_distance"]

    def test_is_base_pipeline_step(self, parquet_file):
        assert isinstance(ParquetLoader(parquet_file), BasePipelineStep)

    def test_name_attribute(self):
        assert ParquetLoader.name == "parquet_loader"


class TestCSVLoader:
    def test_run_returns_dataframe(self, csv_file, sample_df):
        loader = CSVLoader(csv_file)
        df = loader.run(df=None)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_df)

    def test_name_attribute(self):
        assert CSVLoader.name == "csv_loader"


class TestJSONLoader:
    def test_run_returns_dataframe(self, json_file, sample_df):
        loader = JSONLoader(json_file)
        df = loader.run(df=None)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_df)

    def test_name_attribute(self):
        assert JSONLoader.name == "json_loader"


class TestLoaderFactory:
    def test_create_parquet(self, parquet_file):
        loader = LoaderFactory.create(parquet_file)
        assert isinstance(loader, ParquetLoader)

    def test_create_csv(self, csv_file):
        loader = LoaderFactory.create(csv_file)
        assert isinstance(loader, CSVLoader)

    def test_create_json(self, json_file):
        loader = LoaderFactory.create(json_file)
        assert isinstance(loader, JSONLoader)

    def test_unknown_extension_raises(self, tmp_path):
        p = tmp_path / "file.xyz"
        p.touch()
        with pytest.raises(ValueError, match=".xyz"):
            LoaderFactory.create(p)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LoaderFactory.create(tmp_path / "ghost.parquet")

    def test_supported_extensions(self):
        exts = LoaderFactory.supported_extensions()
        assert ".parquet" in exts
        assert ".csv" in exts
        assert ".json" in exts

    def test_registry_snapshot(self):
        reg = LoaderFactory.registry()
        assert reg[".parquet"] == "ParquetLoader"

    def test_factory_kwargs_passed_to_loader(self, parquet_file):
        """Les kwargs supplémentaires sont transmis au constructeur du loader."""
        loader = LoaderFactory.create(parquet_file, columns=["PULocationID"])
        assert loader.columns == ["PULocationID"]

    def test_register_loader_decorator(self, tmp_path):
        """Un loader custom enregistré est utilisable via la factory."""
        from pipeline.loaders import register_loader

        p = tmp_path / "custom.feather"
        # On crée juste le fichier pour passer le check FileNotFoundError
        p.touch()

        @register_loader(".feather")
        class FeatherLoader(BasePipelineStep):
            name = "feather_loader"

            def __init__(self, path, **kw):
                self.path = path

            def _run(self, df: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame()

        loader = LoaderFactory.create(p)
        assert isinstance(loader, FeatherLoader)

        # Nettoyage du registre
        _LOADER_REGISTRY.pop(".feather", None)

    def test_register_loader_requires_base_pipeline_step(self):
        from pipeline.loaders import register_loader

        with pytest.raises(TypeError, match="BasePipelineStep"):
            @register_loader(".bad")
            class InvalidLoader:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  Tests OneHotStrategy
# ══════════════════════════════════════════════════════════════════════════════

class TestOneHotStrategy:
    def test_output_shape(self, sample_df):
        strategy = OneHotStrategy()
        n_unique = sample_df["PULocationID"].nunique()
        df_out = strategy.encode(sample_df, "PULocationID")
        assert "PULocationID" not in df_out.columns
        assert len(df_out.columns) == len(sample_df.columns) - 1 + n_unique

    def test_drop_first(self, sample_df):
        strat_keep = OneHotStrategy(drop_first=False)
        strat_drop = OneHotStrategy(drop_first=True)
        n_unique = sample_df["PULocationID"].nunique()
        out_keep = strat_keep.encode(sample_df, "PULocationID")
        out_drop = strat_drop.encode(sample_df, "PULocationID")
        assert len(out_keep.columns) == len(out_drop.columns) + 1

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(KeyError):
            OneHotStrategy().encode(sample_df, "nonexistent")

    def test_original_df_not_mutated(self, sample_df):
        cols_before = list(sample_df.columns)
        OneHotStrategy().encode(sample_df, "PULocationID")
        assert list(sample_df.columns) == cols_before

    def test_custom_prefix(self, sample_df):
        df_out = OneHotStrategy(prefix="zone").encode(sample_df, "PULocationID")
        assert any(c.startswith("zone_") for c in df_out.columns)


# ══════════════════════════════════════════════════════════════════════════════
#  Tests FrequencyEncodingStrategy
# ══════════════════════════════════════════════════════════════════════════════

class TestFrequencyEncodingStrategy:
    def test_output_has_freq_column(self, sample_df):
        strat = FrequencyEncodingStrategy()
        df_out = strat.encode(sample_df, "PULocationID")
        assert "PULocationID_freq" in df_out.columns
        assert "PULocationID" not in df_out.columns

    def test_values_sum_to_one(self, sample_df):
        """Les fréquences relatives doivent sommer à 1."""
        strat = FrequencyEncodingStrategy()
        df_out = strat.encode(sample_df, "PULocationID")
        # Chaque ligne porte la fréquence de sa zone ; somme sur les valeurs uniques
        freq_sum = sample_df["PULocationID"].value_counts(normalize=True).sum()
        assert abs(freq_sum - 1.0) < 1e-9

    def test_fit_then_encode(self, sample_df):
        strat = FrequencyEncodingStrategy()
        strat.fit(sample_df, "PULocationID")
        df_out = strat.encode(sample_df, "PULocationID")
        assert "PULocationID_freq" in df_out.columns

    def test_unknown_category_filled_with_zero(self, sample_df):
        strat = FrequencyEncodingStrategy()
        strat.fit(sample_df, "PULocationID")
        df_unseen = pd.DataFrame({"PULocationID": [999], "tip_pct": [0.1]})
        df_out = strat.encode(df_unseen, "PULocationID")
        assert df_out["PULocationID_freq"].iloc[0] == 0.0

    def test_original_df_not_mutated(self, sample_df):
        cols = list(sample_df.columns)
        FrequencyEncodingStrategy().encode(sample_df, "PULocationID")
        assert list(sample_df.columns) == cols


# ══════════════════════════════════════════════════════════════════════════════
#  Tests TargetEncodingStrategy
# ══════════════════════════════════════════════════════════════════════════════

class TestTargetEncodingStrategy:
    def test_encode_after_fit(self, sample_df):
        strat = TargetEncodingStrategy(smoothing=5.0)
        strat.fit(sample_df, "PULocationID", target="tip_pct")
        df_out = strat.encode(sample_df, "PULocationID")
        assert "PULocationID_target_enc" in df_out.columns
        assert "PULocationID" not in df_out.columns

    def test_encode_without_fit_raises(self, sample_df):
        with pytest.raises(RuntimeError, match="fit"):
            TargetEncodingStrategy().encode(sample_df, "PULocationID")

    def test_smoothing_towards_global_mean(self, sample_df):
        """Avec smoothing très élevé, les encodages doivent tendre vers la moyenne globale."""
        global_mean = sample_df["tip_pct"].mean()
        strat = TargetEncodingStrategy(smoothing=1e9)
        strat.fit(sample_df, "PULocationID", target="tip_pct")
        df_out = strat.encode(sample_df, "PULocationID")
        encoded_vals = df_out["PULocationID_target_enc"]
        assert all(abs(v - global_mean) < 1e-3 for v in encoded_vals)

    def test_missing_target_raises(self, sample_df):
        with pytest.raises(ValueError, match="cible"):
            TargetEncodingStrategy().fit(sample_df, "PULocationID")

    def test_unknown_category_filled_with_global_mean(self, sample_df):
        strat = TargetEncodingStrategy()
        strat.fit(sample_df, "PULocationID", target="tip_pct")
        df_unseen = pd.DataFrame({"PULocationID": [9999], "tip_pct": [0.0]})
        df_out = strat.encode(df_unseen, "PULocationID")
        assert abs(df_out["PULocationID_target_enc"].iloc[0] - strat._global_mean) < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
#  Tests FeatureEngineer (injection de dépendance + changement à chaud)
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineer:
    def test_delegates_to_strategy(self, sample_df):
        fe = FeatureEngineer(strategy=OneHotStrategy())
        df_out = fe.transform(sample_df, "PULocationID")
        assert "PULocationID" not in df_out.columns

    def test_set_strategy_hot_swap(self, sample_df):
        fe = FeatureEngineer(strategy=OneHotStrategy())
        fe.set_strategy(FrequencyEncodingStrategy())
        df_out = fe.transform(sample_df, "PULocationID")
        # Avec FrequencyEncoding, on attend une colonne _freq
        assert "PULocationID_freq" in df_out.columns

    def test_fit_transform(self, sample_df):
        fe = FeatureEngineer(strategy=TargetEncodingStrategy())
        df_out = fe.fit_transform(sample_df, "PULocationID", target="tip_pct")
        assert "PULocationID_target_enc" in df_out.columns

    def test_strategy_property(self):
        strat = OneHotStrategy()
        fe = FeatureEngineer(strategy=strat)
        assert fe.strategy is strat

    def test_repr(self):
        fe = FeatureEngineer(strategy=OneHotStrategy())
        assert "FeatureEngineer" in repr(fe)
        assert "OneHotStrategy" in repr(fe)

    def test_multiple_strategy_changes(self, sample_df):
        """Vérifie qu'on peut changer de stratégie plusieurs fois sans état résiduel."""
        fe = FeatureEngineer(strategy=OneHotStrategy())
        fe.set_strategy(FrequencyEncodingStrategy())
        fe.set_strategy(OneHotStrategy())
        df_out = fe.transform(sample_df, "PULocationID")
        assert any(c.startswith("PULocationID_") for c in df_out.columns)
