import numpy as np
import pytest

from eicu_kg.metrics import ranking_metrics, true_ranks


def test_optimistic_tie_rank() -> None:
    scores = np.array([[0.7, 0.7, 0.2], [0.1, 0.5, 0.9]])
    targets = np.array([1, 1])
    assert true_ranks(scores, targets).tolist() == [1, 2]


def test_reference_hit_arithmetic() -> None:
    assert 3048 / 4939 == pytest.approx(0.6171289734764123)
    assert 46 / 1886 == pytest.approx(0.024390243902439025)
    assert 3062 / 4939 == pytest.approx(0.6199635553755821)
    assert 56 / 1886 == pytest.approx(0.029692470837751856)


def test_metrics_count_and_rate() -> None:
    scores = np.eye(4)
    targets = np.arange(4)
    metrics = ranking_metrics(scores, targets, k=1)
    assert metrics == {
        "n_queries": 4,
        "mrr": 1.0,
        "hits_at_1": 1.0,
        "hit_count_at_1": 4,
    }
