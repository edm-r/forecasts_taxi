"""
tests/test_config.py
---------------------
Tests unitaires pour les deux implémentations du Singleton (exercice 1.3).

pytest tests/test_config.py -v
"""

import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.config import Config, ConfigNew, SingletonMeta


def _reset_config():
    """Réinitialise le Singleton entre les tests."""
    SingletonMeta._instances.pop(Config, None)

def _reset_config_new():
    ConfigNew._instance = None


# ══════════════════════════════════════════════════════════════════════════════
#  Méthode B — SingletonMeta (Config)
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigSingleton:
    def setup_method(self):
        _reset_config()

    def test_same_instance(self):
        c1, c2 = Config(), Config()
        assert c1 is c2

    def test_identity_after_reset(self):
        c1 = Config()
        _reset_config()
        c2 = Config()
        assert c1 is not c2

    def test_raw_data_path_default(self):
        cfg = Config()
        assert "raw" in str(cfg.raw_data_path)

    def test_processed_data_path_default(self):
        cfg = Config()
        assert "processed" in str(cfg.processed_data_path)

    def test_seed_is_int(self):
        cfg = Config()
        assert isinstance(cfg.seed, int)

    def test_model_returns_dict(self):
        cfg = Config()
        assert isinstance(cfg.model, dict)
        assert "learning_rate" in cfg.model

    def test_pipeline_returns_dict(self):
        cfg = Config()
        assert isinstance(cfg.pipeline, dict)

    # ── Lecture seule ────────────────────────────────────────────────────

    def test_raw_data_path_read_only(self):
        with pytest.raises(AttributeError):
            Config().raw_data_path = "autre/chemin"

    def test_seed_read_only(self):
        with pytest.raises(AttributeError):
            Config().seed = 0

    def test_model_read_only(self):
        with pytest.raises(AttributeError):
            Config().model = {}

    def test_processed_data_path_read_only(self):
        with pytest.raises(AttributeError):
            Config().processed_data_path = "x"

    def test_features_data_path_read_only(self):
        with pytest.raises(AttributeError):
            Config().features_data_path = "x"

    def test_repr(self):
        r = repr(Config())
        assert "Config" in r

    def test_model_dict_is_copy(self):
        """Modifier le dict retourné ne doit pas altérer l'état interne."""
        cfg = Config()
        d = cfg.model
        d["learning_rate"] = 999
        assert cfg.model["learning_rate"] != 999

    def test_loads_yaml_file(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "\n".join(
                [
                    "paths:",
                    "  raw_data: custom/raw",
                    "model:",
                    "  seed: 123",
                    "pipeline:",
                    "  chunk_size: 2048",
                ]
            ),
            encoding="utf-8",
        )

        cfg = Config(config_path=cfg_path)
        assert str(cfg.raw_data_path) == "custom/raw"
        assert cfg.seed == 123
        assert cfg.pipeline["chunk_size"] == 2048


# ══════════════════════════════════════════════════════════════════════════════
#  Méthode A — __new__ (ConfigNew)
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigNewSingleton:
    def setup_method(self):
        _reset_config_new()

    def test_same_instance(self):
        c1, c2 = ConfigNew(), ConfigNew()
        assert c1 is c2

    def test_identity_after_reset(self):
        c1 = ConfigNew()
        _reset_config_new()
        c2 = ConfigNew()
        assert c1 is not c2

    def test_init_not_called_twice(self):
        """L'état ne doit pas être écrasé par un second appel à ConfigNew()."""
        c1 = ConfigNew()
        original_data = id(c1._data)
        _ = ConfigNew()
        assert id(c1._data) == original_data

    def test_raw_data_path_read_only(self):
        with pytest.raises(AttributeError):
            ConfigNew().raw_data_path = "x"

    def test_seed_is_int(self):
        assert isinstance(ConfigNew().seed, int)
