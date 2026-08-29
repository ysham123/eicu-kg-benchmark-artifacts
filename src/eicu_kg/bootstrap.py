"""Patient-cluster bootstrap for query-level metric differences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class BootstrapResult:
    """Point difference and percentile interval from patient resampling."""

    point_difference: float
    lower: float
    upper: float
    n_patients: int
    n_replicates: int
    seed: int
    replicate_differences: NDArray[np.float64]


def patient_cluster_bootstrap_difference(
    parent_hits: ArrayLike,
    candidate_hits: ArrayLike,
    patient_ids: ArrayLike,
    *,
    n_replicates: int = 10_000,
    seed: int = 20_260_828,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Bootstrap a seed-averaged Hits difference by resampling patients.

    ``parent_hits`` has one Boolean value per query. ``candidate_hits`` may be
    one aligned vector or a ``[n_seeds, n_queries]`` matrix; seed values are
    averaged before cluster resampling. Every sampled patient contributes all
    of that patient's queries, preserving within-patient clustering.
    """

    parent = np.asarray(parent_hits)
    candidate = np.asarray(candidate_hits)
    patients = np.asarray(patient_ids)
    if parent.ndim != 1 or patients.shape != parent.shape:
        raise ValueError("parent_hits and patient_ids must be aligned vectors")
    if candidate.ndim == 1:
        candidate = candidate[None, :]
    if candidate.ndim != 2 or candidate.shape[1] != parent.size:
        raise ValueError("candidate_hits must align with the query dimension")
    if parent.size == 0:
        raise ValueError("at least one query is required")
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    parent = parent.astype(np.float64)
    candidate_mean = candidate.astype(np.float64).mean(axis=0)
    query_differences = candidate_mean - parent
    unique_patients, inverse = np.unique(patients, return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(unique_patients.size)]

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_replicates, dtype=np.float64)
    for replicate in range(n_replicates):
        sampled = rng.integers(0, len(groups), size=len(groups))
        numerator = sum(
            float(query_differences[groups[index]].sum()) for index in sampled
        )
        denominator = sum(int(groups[index].size) for index in sampled)
        replicates[replicate] = numerator / denominator

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(replicates, [alpha, 1.0 - alpha])
    return BootstrapResult(
        point_difference=float(query_differences.mean()),
        lower=float(lower),
        upper=float(upper),
        n_patients=int(unique_patients.size),
        n_replicates=n_replicates,
        seed=seed,
        replicate_differences=replicates,
    )
