"""Tests for the EfficientNetB3 fine-label transfer-learning model builder."""

import pytest
import tensorflow as tf

from models.efficientnet_b3_fine import build_efficientnet_b3


def _backbone(model: tf.keras.Model) -> tf.keras.Model:
    return [layer for layer in model.layers if isinstance(layer, tf.keras.Model)][0]


def test_efficientnet_b3_multiclass_shape():
    model = build_efficientnet_b3(num_classes=100, resize_to=64, weights=None)

    assert model.name == "efficientnet_b3_transfer"
    assert model.input_shape == (None, 32, 32, 3)
    assert model.output_shape == (None, 100)


def test_efficientnet_b3_binary_head_shape():
    model = build_efficientnet_b3(num_classes=1, resize_to=64, weights=None)

    assert model.output_shape == (None, 1)
    assert model.layers[-1].activation == tf.keras.activations.sigmoid


def test_efficientnet_b3_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="num_classes"):
        build_efficientnet_b3(num_classes=0, resize_to=64, weights=None)
    with pytest.raises(ValueError, match="resize_to"):
        build_efficientnet_b3(num_classes=100, resize_to=31, weights=None)
    with pytest.raises(ValueError, match="unfreeze_from"):
        build_efficientnet_b3(
            num_classes=100,
            resize_to=64,
            weights=None,
            trainable_backbone=True,
            unfreeze_from="block8",
        )


def test_efficientnet_b3_freezes_backbone_by_default():
    model = build_efficientnet_b3(num_classes=100, resize_to=64, weights=None)

    assert not _backbone(model).trainable


def test_efficientnet_b3_supports_partial_unfreeze_with_frozen_batchnorm():
    model = build_efficientnet_b3(
        num_classes=100,
        resize_to=64,
        weights=None,
        trainable_backbone=True,
        unfreeze_from="block6",
        freeze_bn=True,
    )

    backbone = _backbone(model)
    block5_layers = [layer for layer in backbone.layers if layer.name.startswith("block5")]
    block6_layers = [layer for layer in backbone.layers if layer.name.startswith("block6")]
    batchnorm_layers = [
        layer
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]

    assert block5_layers
    assert block6_layers
    assert all(not layer.trainable for layer in block5_layers)
    assert any(layer.trainable for layer in block6_layers)
    assert all(not layer.trainable for layer in batchnorm_layers)


def test_efficientnet_b3_includes_augmentation_when_enabled():
    model = build_efficientnet_b3(
        num_classes=100,
        resize_to=64,
        weights=None,
        augmentation={"enabled": True, "horizontal_flip": True},
    )

    assert "augmentation" in [layer.name for layer in model.layers]
