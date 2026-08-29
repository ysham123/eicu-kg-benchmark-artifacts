from pathlib import Path

import yaml


def test_fixed_protocol_constants() -> None:
    protocol = yaml.safe_load(Path("protocol.yaml").read_text(encoding="utf-8"))
    assert protocol["benchmark"]["candidate_space_size"] == 147
    assert protocol["hard_slice"]["cutoff"] == 383
    assert protocol["hard_slice"]["test_queries"] == 1886
    csr = protocol["csr_hard"]
    assert csr["inference"] == {
        "budget_q": 0.07,
        "selected_hard_queries": 132,
        "selection_count": "int(round(q * n_hard))",
        "shortlist_m": 25,
        "shortlist_membership": "numpy_argpartition_on_frozen_candidate_order",
        "candidate_order_hash_required": True,
        "fusion_gamma": 1.0,
        "modify_hard_rows_only": True,
        "modify_shortlist_candidates_only": True,
    }
    assert csr["gate"]["feature_order"][-1] == "constant_zero"
    assert csr["gate"]["temperature_scaling"] is False
    assert (
        protocol["diagnostics"]["dxpop_temperature_scaling"]["enters_csr_gate"] is False
    )
