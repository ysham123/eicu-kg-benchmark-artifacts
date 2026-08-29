"""Training-only construction of the eICU-KG HARD slice."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def hard_mask_from_training_counts(
    targets: ArrayLike,
    training_counts: ArrayLike,
    *,
    cutoff: int = 383,
) -> NDArray[np.bool_]:
    """Mark queries whose true concept occurs at most ``cutoff`` in TRAIN."""

    target_array = np.asarray(targets)
    count_array = np.asarray(training_counts)
    if target_array.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if count_array.ndim != 1:
        raise ValueError("training_counts must be one-dimensional")
    if not np.issubdtype(target_array.dtype, np.integer):
        raise TypeError("targets must contain integer candidate indices")
    if np.any(target_array < 0) or np.any(target_array >= count_array.size):
        raise ValueError("target index is outside training_counts")
    if np.any(count_array < 0):
        raise ValueError("training counts cannot be negative")
    return count_array[target_array.astype(np.int64, copy=False)] <= cutoff


def assert_hard_partition(mask: ArrayLike, *, expected_size: int | None = None) -> None:
    """Validate a one-dimensional Boolean HARD mask."""

    mask_array = np.asarray(mask)
    if mask_array.ndim != 1 or mask_array.dtype != np.bool_:
        raise TypeError("HARD mask must be a one-dimensional Boolean array")
    if expected_size is not None and mask_array.size != expected_size:
        raise ValueError(
            f"HARD mask has {mask_array.size} rows; expected {expected_size}"
        )
