"""Fixed-protocol utilities for the CSR-HARD benchmark-side case study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # Public metric checks do not require PyTorch.
    torch = None
    nn = None

from .metrics import ranking_metrics, true_ranks

GATE_ARCHITECTURE = (8, 64, 64, 1)


def _score_pair(dx_scores: ArrayLike, rg_scores: ArrayLike) -> tuple[NDArray, NDArray]:
    dx = np.asarray(dx_scores)
    rg = np.asarray(rg_scores)
    if dx.ndim != 2 or rg.shape != dx.shape:
        raise ValueError("DXPOP and RGCN scores must share shape [n, k]")
    if dx.shape[1] < 2:
        raise ValueError("at least two candidates are required")
    if not np.all(np.isfinite(dx)) or not np.all(np.isfinite(rg)):
        raise ValueError("score matrices contain non-finite values")
    return dx, rg


def row_entropy(scores: ArrayLike, *, epsilon: float = 1.0e-9) -> NDArray:
    """Compute entropy after row-wise max-stable softmax."""

    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("scores must be two-dimensional")
    shifted = array - np.max(array, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return -np.sum(probabilities * np.log(probabilities + epsilon), axis=1)


def build_gate_features(dx_scores: ArrayLike, rg_scores: ArrayLike) -> NDArray:
    """Build the exact eight serialized gate coordinates.

    The first four coordinates are raw top-1 logits and raw top-1/top-2
    margins. Agreement is binary, entropies are row-wise, and the final
    coordinate is the constant zero retained by the released architecture.
    No temperature scaling or cross-query normalization is applied.
    """

    dx, rg = _score_pair(dx_scores, rg_scores)
    dx_sorted = np.sort(dx, axis=1)
    rg_sorted = np.sort(rg, axis=1)
    dx_top1 = dx_sorted[:, -1]
    rg_top1 = rg_sorted[:, -1]
    dx_margin = dx_top1 - dx_sorted[:, -2]
    rg_margin = rg_top1 - rg_sorted[:, -2]
    agree = (np.argmax(dx, axis=1) == np.argmax(rg, axis=1)).astype(np.float64)
    zeros = np.zeros(dx.shape[0], dtype=np.float64)
    return np.column_stack(
        [
            dx_top1,
            dx_margin,
            rg_top1,
            rg_margin,
            agree,
            row_entropy(dx),
            row_entropy(rg),
            zeros,
        ]
    )


def shortlist_fusion(
    dx_scores: ArrayLike,
    rg_scores: ArrayLike,
    *,
    shortlist_m: int,
    gamma: float,
    selected_rows: ArrayLike | None = None,
) -> NDArray:
    """Add RGCN scores only to each selected row's DXPOP top-M candidates."""

    dx, rg = _score_pair(dx_scores, rg_scores)
    if not 1 <= shortlist_m <= dx.shape[1]:
        raise ValueError("shortlist_m is outside the candidate space")
    if selected_rows is None:
        selected = np.ones(dx.shape[0], dtype=bool)
    else:
        selected = np.asarray(selected_rows)
        if selected.shape != (dx.shape[0],) or selected.dtype != np.bool_:
            raise TypeError("selected_rows must be a Boolean vector with one row flag")

    fused = dx.copy()
    rows = np.flatnonzero(selected)
    for row in rows:
        # Preserve the released implementation. Candidate order and NumPy
        # version must be recorded because argpartition does not define a
        # secondary order at an exact tie boundary.
        order = np.argpartition(dx[row], -shortlist_m)[-shortlist_m:]
        fused[row, order] = dx[row, order] + gamma * rg[row, order]
    return fused


def select_hard_rows(
    gate_scores: ArrayLike,
    hard_mask: ArrayLike,
    *,
    budget_q: float,
) -> NDArray[np.bool_]:
    """Select the highest gate scores among HARD rows under a fixed budget."""

    gate = np.asarray(gate_scores)
    hard = np.asarray(hard_mask)
    if gate.ndim != 1 or hard.shape != gate.shape or hard.dtype != np.bool_:
        raise TypeError("gate_scores and Boolean hard_mask must be aligned vectors")
    if not 0.0 <= budget_q <= 1.0:
        raise ValueError("budget_q must lie in [0, 1]")
    if not np.all(np.isfinite(gate)):
        raise ValueError("gate_scores contain a non-finite value")

    hard_rows = np.flatnonzero(hard)
    n_select = int(round(budget_q * hard_rows.size))  # noqa: RUF046
    ranked = hard_rows[np.argsort(-gate[hard_rows], kind="stable")]
    selected = np.zeros(gate.size, dtype=bool)
    selected[ranked[:n_select]] = True
    return selected


def apply_csr_hard(
    dx_scores: ArrayLike,
    rg_scores: ArrayLike,
    gate_scores: ArrayLike,
    hard_mask: ArrayLike,
    *,
    budget_q: float = 0.07,
    shortlist_m: int = 25,
    gamma: float = 1.0,
) -> tuple[NDArray, NDArray[np.bool_]]:
    """Apply the fixed CSR-HARD inference transformation."""

    selected = select_hard_rows(gate_scores, hard_mask, budget_q=budget_q)
    fused = shortlist_fusion(
        dx_scores,
        rg_scores,
        shortlist_m=shortlist_m,
        gamma=gamma,
        selected_rows=selected,
    )
    return fused, selected


def make_gate_labels(
    dx_scores: ArrayLike,
    rg_scores: ArrayLike,
    targets: ArrayLike,
    hard_mask: ArrayLike,
    *,
    shortlist_m: int = 25,
    gamma: float = 2.0,
    top_k: int = 10,
) -> tuple[NDArray[np.float32], bool]:
    """Create TRAIN-only gate labels and the documented empty-positive fallback."""

    hard = np.asarray(hard_mask)
    dx, _ = _score_pair(dx_scores, rg_scores)
    if hard.shape != (dx.shape[0],) or hard.dtype != np.bool_:
        raise TypeError("hard_mask must be a Boolean vector aligned to TRAIN")
    r0 = true_ranks(dx_scores, targets)
    reference = shortlist_fusion(
        dx_scores, rg_scores, shortlist_m=shortlist_m, gamma=gamma
    )
    r1 = true_ranks(reference, targets)
    labels = hard & (r0 > top_k) & (r1 <= top_k)
    fallback_used = not np.any(labels)
    if fallback_used:
        labels = hard & ((r0 - r1) >= 1)
    if not np.any(labels):
        raise ValueError("gate-label construction produced no positive examples")
    return labels.astype(np.float32), fallback_used


if nn is not None:

    class GateMLP(nn.Module):
        """Released 8-to-64-to-64-to-1 gate architecture."""

        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.network(features).squeeze(-1)

else:

    class GateMLP:  # pragma: no cover - exercised only without the train extra.
        """Placeholder that explains how to enable gate training."""

        def __init__(self) -> None:
            raise ImportError("Install the 'train' extra to construct GateMLP")


@dataclass(frozen=True)
class GateTrainingRecord:
    """Auditable facts from one fixed gate-training run."""

    seed: int
    epochs: int
    positive_count: int
    negative_count: int
    fallback_used: bool
    final_loss: float


def train_gate(
    dx_scores: ArrayLike,
    rg_scores: ArrayLike,
    targets: ArrayLike,
    hard_mask: ArrayLike,
    *,
    seed: int,
    epochs: int = 40,
    learning_rate: float = 2.0e-3,
    weight_decay: float = 1.0e-4,
) -> tuple[GateMLP, GateTrainingRecord]:
    """Train the gate for a fixed number of full-batch AdamW epochs."""

    if torch is None or nn is None:
        raise ImportError("Install the 'train' extra to train the gate")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    features = build_gate_features(dx_scores, rg_scores)
    labels, fallback_used = make_gate_labels(dx_scores, rg_scores, targets, hard_mask)
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if negative_count == 0:
        raise ValueError("gate labels contain no negative examples")

    torch.manual_seed(seed)
    model = GateMLP()
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    pos_weight = torch.tensor(negative_count / positive_count, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    final_loss = float("nan")
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    return model, GateTrainingRecord(
        seed=seed,
        epochs=epochs,
        positive_count=positive_count,
        negative_count=negative_count,
        fallback_used=fallback_used,
        final_loss=final_loss,
    )


def validation_guard(
    parent_scores: ArrayLike,
    candidate_scores: ArrayLike,
    targets: ArrayLike,
    *,
    epsilon: float = 0.001,
) -> tuple[bool, dict[str, float]]:
    """Apply the two-sided full-validation aggregate-stability guard."""

    parent = ranking_metrics(parent_scores, targets, k=10)
    candidate = ranking_metrics(candidate_scores, targets, k=10)
    delta_mrr = float(candidate["mrr"] - parent["mrr"])
    delta_h10 = float(candidate["hits_at_10"] - parent["hits_at_10"])
    accepted = abs(delta_mrr) <= epsilon and abs(delta_h10) <= epsilon
    return accepted, {"delta_mrr": delta_mrr, "delta_hits_at_10": delta_h10}
