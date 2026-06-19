"""Loss functions for CIFAR-100 binary (and future multiclass) tasks."""

import tensorflow as tf

BinaryCrossentropy = tf.keras.losses.BinaryCrossentropy
Loss = tf.keras.losses.Loss


def get_loss(head: str = "binary") -> Loss:
    """Return the Keras loss for the chosen output head.

    Only the binary head is supported on the baseline branch; the multiclass
    path will be added when a multiclass head is actually introduced.
    """
    if head == "binary":
        return BinaryCrossentropy(from_logits=False)
    raise ValueError(
        f"head must be 'binary' (multiclass not yet supported); got {head!r}"
    )
