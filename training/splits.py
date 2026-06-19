"""Deterministic stratified train/validation split for binary tasks."""

import numpy as np


def stratified_train_val_split(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split images/labels into train and validation, stratified by label.

    Stratification preserves the positive/negative ratio in both splits, so
    binary metrics on the validation set remain meaningful even when the
    positive class is rare.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must be in (0, 1); got {val_fraction}"
        )
    labels = np.asarray(labels).reshape(-1)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"images and labels length mismatch: "
            f"{images.shape[0]} vs {labels.shape[0]}"
        )

    rng = np.random.default_rng(seed)
    train_idx_parts: list[np.ndarray] = []
    val_idx_parts: list[np.ndarray] = []
    for cls in (0, 1):
        cls_idx = np.flatnonzero(labels == cls)
        rng.shuffle(cls_idx)
        n_val = int(round(cls_idx.size * val_fraction))
        if cls_idx.size > 1:
            n_val = max(1, min(n_val, cls_idx.size - 1))
        val_idx_parts.append(cls_idx[:n_val])
        train_idx_parts.append(cls_idx[n_val:])

    train_idx = np.concatenate(train_idx_parts)
    val_idx = np.concatenate(val_idx_parts)
    # Reshuffle so positives/negatives are interleaved.
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return (
        images[train_idx],
        labels[train_idx],
        images[val_idx],
        labels[val_idx],
    )
