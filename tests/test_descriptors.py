"""
tests/test_descriptors.py
--------------------------
Tests unitaires pour les descripteurs typés (exercice 1.2).

pytest tests/test_descriptors.py -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.descriptors import (
    BoundedFloat, OneOf, Positive, TypedAttr, ModelConfig,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Positive
# ══════════════════════════════════════════════════════════════════════════════

class TestPositive:
    def setup_method(self):
        class Cfg:
            val = Positive(min_value=10)
        self.Cfg = Cfg

    def test_valid_value(self):
        c = self.Cfg(); c.val = 11
        assert c.val == 11

    def test_equal_to_min_raises(self):
        c = self.Cfg()
        with pytest.raises(ValueError, match="10"):
            c.val = 10

    def test_below_min_raises(self):
        c = self.Cfg()
        with pytest.raises(ValueError):
            c.val = 5

    def test_float_accepted(self):
        c = self.Cfg(); c.val = 10.5
        assert c.val == 10.5

    def test_wrong_type_raises(self):
        c = self.Cfg()
        with pytest.raises(TypeError):
            c.val = "beaucoup"

    def test_default_min_value(self):
        class Cfg2:
            val = Positive()
        c = Cfg2(); c.val = 0.001
        assert c.val == 0.001

    def test_zero_raises_with_default(self):
        class Cfg2:
            val = Positive()
        with pytest.raises(ValueError):
            Cfg2().val = 0

    def test_independence_between_instances(self):
        c1, c2 = self.Cfg(), self.Cfg()
        c1.val = 50; c2.val = 100
        assert c1.val == 50
        assert c2.val == 100


# ══════════════════════════════════════════════════════════════════════════════
#  OneOf
# ══════════════════════════════════════════════════════════════════════════════

class TestOneOf:
    def setup_method(self):
        class Cfg:
            mode = OneOf("a", "b", "c")
        self.Cfg = Cfg

    def test_valid_choice(self):
        c = self.Cfg(); c.mode = "b"
        assert c.mode == "b"

    def test_invalid_choice_raises(self):
        with pytest.raises(ValueError, match="'d'"):
            self.Cfg().mode = "d"

    def test_no_choices_raises(self):
        with pytest.raises(ValueError):
            OneOf()

    def test_works_with_numbers(self):
        class Cfg2:
            n = OneOf(1, 2, 3)
        c = Cfg2(); c.n = 2
        assert c.n == 2


# ══════════════════════════════════════════════════════════════════════════════
#  TypedAttr
# ══════════════════════════════════════════════════════════════════════════════

class TestTypedAttr:
    def setup_method(self):
        class Cfg:
            depth = TypedAttr(int)
        self.Cfg = Cfg

    def test_valid_type(self):
        c = self.Cfg(); c.depth = 6
        assert c.depth == 6

    def test_float_rejected(self):
        with pytest.raises(TypeError, match="int"):
            self.Cfg().depth = 6.0

    def test_string_rejected(self):
        with pytest.raises(TypeError):
            self.Cfg().depth = "six"

    def test_bool_rejected_with_strict_type(self):
        with pytest.raises(TypeError, match="int"):
            self.Cfg().depth = True


# ══════════════════════════════════════════════════════════════════════════════
#  BoundedFloat
# ══════════════════════════════════════════════════════════════════════════════

class TestBoundedFloat:
    def setup_method(self):
        class Cfg:
            lr = BoundedFloat(0.0, 1.0)
        self.Cfg = Cfg

    def test_valid_mid(self):
        c = self.Cfg(); c.lr = 0.5
        assert c.lr == 0.5

    def test_boundary_low(self):
        c = self.Cfg(); c.lr = 0.0
        assert c.lr == 0.0

    def test_boundary_high(self):
        c = self.Cfg(); c.lr = 1.0
        assert c.lr == 1.0

    def test_above_high_raises(self):
        with pytest.raises(ValueError, match="1.0"):
            self.Cfg().lr = 1.001

    def test_below_low_raises(self):
        with pytest.raises(ValueError):
            self.Cfg().lr = -0.01

    def test_string_raises(self):
        with pytest.raises(TypeError):
            self.Cfg().lr = "slow"

    def test_int_rejected(self):
        with pytest.raises(TypeError, match="float"):
            self.Cfg().lr = 1

    def test_inverted_bounds_raises(self):
        with pytest.raises(ValueError):
            BoundedFloat(1.0, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  ModelConfig intégration
# ══════════════════════════════════════════════════════════════════════════════

class TestModelConfig:
    def test_full_valid_config(self):
        cfg = ModelConfig()
        cfg.learning_rate = 0.05
        cfg.n_estimators  = 200
        cfg.objective     = "regression"
        cfg.max_depth     = 8
        assert cfg.learning_rate == 0.05
        assert cfg.n_estimators  == 200
        assert cfg.objective     == "regression"
        assert cfg.max_depth     == 8

    def test_repr(self):
        cfg = ModelConfig()
        cfg.learning_rate = 0.01
        cfg.n_estimators  = 100
        cfg.objective     = "binary"
        cfg.max_depth     = 4
        r = repr(cfg)
        assert "0.01" in r and "binary" in r

    @pytest.mark.parametrize("lr", [-0.1, 1.1, 2.0])
    def test_invalid_learning_rate(self, lr):
        with pytest.raises(ValueError):
            ModelConfig().learning_rate = lr

    def test_invalid_n_estimators(self):
        with pytest.raises(ValueError):
            ModelConfig().n_estimators = 5

    def test_invalid_objective(self):
        with pytest.raises(ValueError):
            ModelConfig().objective = "clustering"

    def test_invalid_max_depth(self):
        with pytest.raises(TypeError):
            ModelConfig().max_depth = 4.5
