"""Tests for the optimizer factory (training.optimizers)."""

import pytest
import tensorflow as tf

from training.optimizers import build_optimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ADAM_CLASSES = (
    tf.keras.optimizers.Adam,
    getattr(getattr(tf.keras.optimizers, "legacy", None), "Adam", tf.keras.optimizers.Adam),
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_optimizer_returns_adam_with_default_lr():
    """Config with only ``optimizer: adam`` should yield Adam at lr=1e-3."""
    opt = build_optimizer({"optimizer": "adam"})
    assert type(opt).__name__ == "Adam"
    assert isinstance(opt, _ADAM_CLASSES)
    lr_value = float(tf.keras.backend.get_value(opt.learning_rate))
    assert lr_value == pytest.approx(1e-3, rel=1e-5)


def test_build_optimizer_honours_learning_rate():
    """Explicit ``learning_rate`` in config must be reflected on the returned optimizer."""
    opt = build_optimizer({"optimizer": "adam", "learning_rate": 1e-4})
    lr_value = float(tf.keras.backend.get_value(opt.learning_rate))
    assert lr_value == pytest.approx(1e-4, rel=1e-5)


def test_build_optimizer_rejects_unknown_name():
    """Unsupported optimizer names must raise ``ValueError``."""
    with pytest.raises(ValueError, match="sgd"):
        build_optimizer({"optimizer": "sgd"})


def test_build_optimizer_default_optimizer_key_is_adam():
    """Omitting ``optimizer`` from config should default to Adam."""
    opt = build_optimizer({})
    assert type(opt).__name__ == "Adam"


def test_build_optimizer_case_insensitive():
    """Optimizer name lookup should be case-insensitive."""
    opt = build_optimizer({"optimizer": "Adam"})
    assert type(opt).__name__ == "Adam"
