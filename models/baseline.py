"""Baseline CNN for CIFAR-100 binary tasks (image view)."""

import tensorflow as tf

keras = tf.keras
layers = tf.keras.layers


def build_baseline_cnn(
    input_shape: tuple[int, int, int] = (32, 32, 3),
    dropout: float = 0.3,
) -> keras.Model:
    """Build a compact 2-block CNN for binary image classification.

    Two conv blocks (32 then 64 filters, 3x3, ReLU), max-pool + dropout after
    each block, then a 128-unit dense head with sigmoid output. Intentionally
    modest so later sequential / attention / transfer models have a fair
    reference point. The model is returned uncompiled; `training.train`
    compiles it with the configured loss and optimizer.
    """
    inputs = keras.Input(shape=input_shape, name="image")

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
