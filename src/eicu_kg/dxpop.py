"""Exact diagnosis-signature frequency prior used by the CSR case study."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def canonical_signature(value: Hashable | Iterable[Hashable]) -> tuple[Hashable, ...]:
    """Return a deterministic set-valued diagnosis signature."""

    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        return tuple(sorted(set(value), key=str))
    except TypeError:
        return (value,)  # type: ignore[arg-type]


def _rank_preserving_scores(
    counts: NDArray[np.int64], global_rank: NDArray[np.int64]
) -> NDArray[np.float64]:
    order = np.lexsort((global_rank, -counts))
    scores = np.empty(counts.size, dtype=np.float64)
    scores[order] = np.arange(counts.size, 0, -1, dtype=np.float64)
    return scores


def signature_prior_scores(
    train_signatures: Sequence[Hashable | Iterable[Hashable]],
    train_targets: ArrayLike,
    query_signatures: Sequence[Hashable | Iterable[Hashable]],
    *,
    n_candidates: int,
) -> NDArray[np.float64]:
    """Score candidates by exact signature counts with global fallback.

    Count ties use global training popularity. The returned values are unique
    rank-preserving scores, so downstream optimistic-rank evaluation is
    deterministic.
    """

    targets = np.asarray(train_targets)
    if len(train_signatures) != targets.size:
        raise ValueError("train_signatures and train_targets differ in length")
    if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("train_targets must be a one-dimensional integer array")
    if n_candidates < 1:
        raise ValueError("n_candidates must be positive")
    if np.any(targets < 0) or np.any(targets >= n_candidates):
        raise ValueError("training target is outside the candidate space")

    global_counts = np.bincount(targets, minlength=n_candidates).astype(np.int64)
    global_order = np.lexsort((np.arange(n_candidates), -global_counts))
    global_rank = np.empty(n_candidates, dtype=np.int64)
    global_rank[global_order] = np.arange(n_candidates)

    by_signature: dict[tuple[Hashable, ...], NDArray[np.int64]] = defaultdict(
        lambda: np.zeros(n_candidates, dtype=np.int64)
    )
    for signature, target in zip(train_signatures, targets, strict=True):
        by_signature[canonical_signature(signature)][int(target)] += 1

    fallback = _rank_preserving_scores(global_counts, global_rank)
    output = np.empty((len(query_signatures), n_candidates), dtype=np.float64)
    for row, signature in enumerate(query_signatures):
        counts = by_signature.get(canonical_signature(signature))
        output[row] = (
            fallback if counts is None else _rank_preserving_scores(counts, global_rank)
        )
    return output
