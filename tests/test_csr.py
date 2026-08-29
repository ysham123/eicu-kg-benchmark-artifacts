import numpy as np
import pytest

from eicu_kg.csr import (
    GATE_ARCHITECTURE,
    apply_csr_hard,
    build_gate_features,
    select_hard_rows,
    shortlist_fusion,
    validation_guard,
)


def test_gate_feature_order_constant_and_row_independence() -> None:
    dx = np.array([[3.0, 1.0, 0.0], [0.0, 4.0, 2.0]])
    rg = np.array([[2.0, 3.0, 0.0], [0.0, 1.0, 5.0]])
    features = build_gate_features(dx, rg)
    assert features.shape == (2, 8)
    assert features[:, :5].tolist() == [
        [3.0, 2.0, 3.0, 1.0, 0.0],
        [4.0, 2.0, 5.0, 4.0, 0.0],
    ]
    assert np.all(features[:, 7] == 0.0)
    assert np.allclose(build_gate_features(dx[::-1], rg[::-1]), features[::-1])


def test_gate_architecture() -> None:
    assert GATE_ARCHITECTURE == (8, 64, 64, 1)


def test_budget_rounding_and_hard_only_selection() -> None:
    gate = np.arange(1886, dtype=float)
    hard = np.ones(1886, dtype=bool)
    selected = select_hard_rows(gate, hard, budget_q=0.07)
    assert selected.sum() == 132


def test_fusion_changes_only_selected_top_m() -> None:
    dx = np.array([[5.0, 4.0, 3.0], [5.0, 4.0, 3.0]])
    rg = np.ones_like(dx)
    selected = np.array([True, False])
    fused = shortlist_fusion(dx, rg, shortlist_m=2, gamma=1.0, selected_rows=selected)
    assert fused[0].tolist() == [6.0, 5.0, 3.0]
    assert np.array_equal(fused[1], dx[1])


def test_apply_csr_leaves_head_rows_exactly_unchanged() -> None:
    dx = np.tile(np.array([3.0, 2.0, 1.0]), (10, 1))
    rg = np.tile(np.array([0.0, 1.0, 2.0]), (10, 1))
    gate = np.arange(10, dtype=float)
    hard = np.array([True] * 5 + [False] * 5)
    fused, selected = apply_csr_hard(
        dx, rg, gate, hard, budget_q=0.4, shortlist_m=2, gamma=1.0
    )
    assert selected.sum() == 2
    assert not np.any(selected[~hard])
    assert np.array_equal(fused[~hard], dx[~hard])


def test_validation_guard_is_two_sided_and_requires_both() -> None:
    parent = np.array([[2.0, 1.0], [2.0, 1.0]])
    targets = np.array([0, 0])
    accepted, delta = validation_guard(parent, parent.copy(), targets)
    assert accepted
    assert delta == {"delta_mrr": 0.0, "delta_hits_at_10": 0.0}

    degraded = np.array([[1.0, 2.0], [1.0, 2.0]])
    accepted, delta = validation_guard(parent, degraded, targets)
    assert not accepted
    assert delta["delta_mrr"] == pytest.approx(-0.5)
