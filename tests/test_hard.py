import numpy as np

from eicu_kg.hard import hard_mask_from_training_counts


def test_hard_uses_inclusive_training_cutoff() -> None:
    counts = np.array([401, 383, 12])
    targets = np.array([0, 1, 2])
    assert hard_mask_from_training_counts(targets, counts).tolist() == [
        False,
        True,
        True,
    ]
