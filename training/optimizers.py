"""Optimizer factory for config-driven training.

Prefers ``tf.keras.optimizers.legacy.Adam`` on macOS / Apple Silicon to avoid
the "Slow optimizer" notice emitted by TensorFlow 2.x when the current-gen
Adam runs on Metal.  Falls back to ``tf.keras.optimizers.Adam`` on TF builds
that do not ship the ``legacy`` namespace.
"""

import tensorflow as tf


def build_optimizer(config: dict) -> tf.keras.optimizers.Optimizer:
    """Return a Keras optimizer built from *config*.

    Reads the following keys:

    - ``optimizer`` (str, default ``"adam"``): name of the optimizer.
      Only ``"adam"`` is supported on this branch; anything else raises
      :class:`ValueError`.
    - ``learning_rate`` (float, default ``1e-3``): initial learning rate.

    The function prefers ``tf.keras.optimizers.legacy.Adam`` when the
    ``legacy`` submodule is available (TF 2.11–2.15 on macOS/M-series),
    and falls back to ``tf.keras.optimizers.Adam`` otherwise.
    """
    name = str(config.get("optimizer", "adam")).lower()
    lr = float(config.get("learning_rate", 1e-3))

    if name == "adam":
        legacy_ns = getattr(tf.keras.optimizers, "legacy", None)
        adam_cls = getattr(legacy_ns, "Adam", None) if legacy_ns is not None else None
        if adam_cls is None:
            adam_cls = tf.keras.optimizers.Adam
        return adam_cls(learning_rate=lr)

    raise ValueError(
        f"Unsupported optimizer {name!r}. "
        "This branch only supports 'adam'."
    )
