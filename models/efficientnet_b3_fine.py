"""EfficientNetB3 transfer-learning model builder for CIFAR-100 fine labels."""

import tensorflow as tf

from models.augmentation import build_augmentation

keras = tf.keras
layers = tf.keras.layers

SUPPORTED_UNFREEZE_POINTS = {None, "all", "block5", "block6", "block7"}


def build_efficientnet_b3(
    input_shape: tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 100,
    dropout: float = 0.4,
    trainable_backbone: bool = False,
    resize_to: int = 160,
    weights: str | None = "imagenet",
    augmentation: dict | None = None,
    unfreeze_from: str | None = None,
    freeze_bn: bool = True,
) -> keras.Model:
    """Build an EfficientNetB3 classifier for CIFAR-100 fine-label transfer.

    The project image pipeline emits float32 CIFAR-100 tensors in ``[0, 1]``.
    Keras EfficientNet contains its own preprocessing layers and expects
    pixel-like values, so the graph resizes and rescales back to ``[0, 255]``
    before the ImageNet backbone.

    ``unfreeze_from`` can partially fine-tune the backbone by layer-name prefix
    (for example, ``"block6"``). BatchNorm layers are kept frozen by default
    and the backbone is called in inference mode to preserve pretrained moving
    statistics during small-batch fine-tuning.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1; got {num_classes}")
    if resize_to < 32:
        raise ValueError(f"resize_to must be >= 32; got {resize_to}")
    if unfreeze_from not in SUPPORTED_UNFREEZE_POINTS:
        supported = ", ".join(sorted(p for p in SUPPORTED_UNFREEZE_POINTS if p))
        raise ValueError(
            f"unfreeze_from must be one of None, {supported}; got {unfreeze_from!r}"
        )

    inputs = keras.Input(shape=input_shape, name="image")

    aug_layer = build_augmentation(augmentation)
    x = aug_layer(inputs) if aug_layer is not None else inputs

    x = layers.Resizing(
        resize_to,
        resize_to,
        interpolation="bilinear",
        name="resize",
    )(x)
    x = layers.Rescaling(255.0, name="rescale_to_0_255")(x)

    backbone = keras.applications.EfficientNetB3(
        include_top=False,
        weights=weights,
        input_shape=(resize_to, resize_to, 3),
        pooling="avg",
    )
    _configure_backbone(
        backbone,
        trainable_backbone=trainable_backbone,
        unfreeze_from=unfreeze_from,
        freeze_bn=freeze_bn,
    )

    x = backbone(x, training=False)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="prob")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="prob")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_b3_transfer")


def _configure_backbone(
    backbone: keras.Model,
    *,
    trainable_backbone: bool,
    unfreeze_from: str | None,
    freeze_bn: bool,
) -> None:
    """Apply frozen, partial, or full trainability to the EfficientNet backbone."""
    if not trainable_backbone:
        backbone.trainable = False
        return

    backbone.trainable = True
    if unfreeze_from not in (None, "all"):
        trainable = False
        for layer in backbone.layers:
            if layer.name.startswith(unfreeze_from):
                trainable = True
            layer.trainable = trainable

    if freeze_bn:
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
