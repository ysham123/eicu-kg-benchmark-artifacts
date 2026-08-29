import numpy as np

from eicu_kg.dxpop import signature_prior_scores
from eicu_kg.metrics import true_ranks


def test_signature_prior_and_global_fallback() -> None:
    train_signatures = [("a",), ("a",), ("b",)]
    train_targets = np.array([1, 1, 2])
    query_signatures = [("a",), ("unseen",)]
    scores = signature_prior_scores(
        train_signatures, train_targets, query_signatures, n_candidates=3
    )
    assert true_ranks(scores, np.array([1, 1])).tolist() == [1, 1]
    assert scores.shape == (2, 3)
