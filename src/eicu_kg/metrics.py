"""Ranking metrics with the tie convention used by eICU-KG."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validated(scores: ArrayLike, targets: ArrayLike) -> tuple[NDArray, NDArray]:
    score_array = np.asarray(scores)
    target_array = np.asarray(targets)

    if score_array.ndim != 2:
        raise ValueError("scores must have shape [n_queries, n_candidates]")
    if target_array.ndim != 1 or target_array.shape[0] != score_array.shape[0]:
        raise ValueError("targets must have shape [n_queries]")
    if not np.issubdtype(target_array.dtype, np.integer):
        raise TypeError("targets must contain integer candidate indices")
    if score_array.shape[1] == 0:
        raise ValueError("scores must contain at least one candidate")
    if np.any(target_array < 0) or np.any(target_array >= score_array.shape[1]):
        raise ValueError("target index is outside the candidate space")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores contain a non-finite value")

    return score_array, target_array.astype(np.int64, copy=False)


def true_ranks(scores: ArrayLike, targets: ArrayLike) -> NDArray[np.int64]:
    """Return optimistic one-based ranks of the true candidate.

    The benchmark rank is ``1 + sum(scores > true_score)``. Candidates tied
    with the true score do not increase its rank.
    """

    score_array, target_array = _validated(scores, targets)
    true_scores = score_array[np.arange(score_array.shape[0]), target_array]
    return 1 + np.sum(score_array > true_scores[:, None], axis=1)


def ranking_metrics(
    scores: ArrayLike,
    targets: ArrayLike,
    *,
    k: int = 10,
) -> dict[str, float | int]:
    """Compute query-level MRR and Hits@``k``."""

    if k < 1:
        raise ValueError("k must be positive")
    ranks = true_ranks(scores, targets)
    return {
        "n_queries": int(ranks.size),
        "mrr": float(np.mean(1.0 / ranks)),
        f"hits_at_{k}": float(np.mean(ranks <= k)),
        f"hit_count_at_{k}": int(np.sum(ranks <= k)),
    }
