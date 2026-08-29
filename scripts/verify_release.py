"""Fail-closed checks for the sanitized public release tree."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eicu_kg.audit import restricted_release_paths


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def load_protocol() -> dict:
    with (ROOT / "protocol.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def verify_protocol(protocol: dict) -> None:
    benchmark = protocol["benchmark"]
    split = protocol["split"]
    hard = protocol["hard_slice"]
    csr = protocol["csr_hard"]
    require(benchmark["candidate_space_size"] == 147, "candidate space changed")
    require(split["target_edges"]["test"] == 4939, "test size changed")
    require(hard["cutoff"] == 383, "HARD cutoff changed")
    require(hard["test_queries"] == 1886, "HARD test size changed")
    require(hard["head_test_queries"] == 3053, "head test size changed")
    require(hard["hard_concepts"] == 137, "HARD concept count changed")
    require(csr["inference"]["budget_q"] == 0.07, "primary q changed")
    require(csr["inference"]["shortlist_m"] == 25, "primary M changed")
    require(csr["inference"]["fusion_gamma"] == 1.0, "primary gamma changed")
    require(csr["gate"]["temperature_scaling"] is False, "gate uses temperature")
    require(csr["gate"]["feature_order"][-1] == "constant_zero", "feature 8 changed")
    guard = csr["validation_guard"]
    require(guard["delta"] == "two_sided_absolute", "guard is not two-sided")
    require(guard["require_both"] is True, "guard does not require both metrics")
    require(guard["mrr_tolerance"] == 0.001, "MRR tolerance changed")
    require(guard["hits_at_10_tolerance"] == 0.001, "H@10 tolerance changed")


def verify_metrics() -> None:
    with (ROOT / "results" / "paper_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_key = {(row["model_id"], row["split"], row["seed"]): row for row in rows}

    signature_test = by_key[("DXPOP-Signature", "test_all", "deterministic")]
    signature_hard = by_key[("DXPOP-Signature", "hard", "deterministic")]
    additive_test = by_key[("DXPOP-Additive", "test_all", "deterministic")]
    additive_hard = by_key[("DXPOP-Additive", "hard", "deterministic")]
    csr_mean = by_key[("CSR-HARD", "hard", "mean")]

    require(
        math.isclose(float(signature_test["hits_at_10"]), 3048 / 4939),
        "DXPOP-Signature test H@10 mismatch",
    )
    require(
        math.isclose(float(signature_hard["hits_at_10"]), 46 / 1886),
        "DXPOP-Signature HARD H@10 mismatch",
    )
    require(
        math.isclose(float(additive_test["hits_at_10"]), 3062 / 4939),
        "DXPOP-Additive test H@10 mismatch",
    )
    require(
        math.isclose(float(additive_hard["hits_at_10"]), 56 / 1886),
        "DXPOP-Additive HARD H@10 mismatch",
    )
    require(
        math.isclose(float(csr_mean["hits_at_10"]), (54 + 63 + 59) / 3 / 1886),
        "CSR fixed-q mean H@10 mismatch",
    )


def verify_release_text() -> None:
    paths = [ROOT / "README.md", ROOT / "DATA.md", ROOT / "protocol.yaml"]
    joined = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()
    stale = ["0.0332", "0.0344", "no-regression", "final_fixed", "patched_official"]
    present = [token for token in stale if token in joined]
    require(not present, f"stale release language found: {present}")


def main() -> None:
    violations = restricted_release_paths(ROOT)
    require(not violations, f"restricted release paths found: {violations}")
    protocol = load_protocol()
    verify_protocol(protocol)
    verify_metrics()
    verify_release_text()
    print("Release verification passed.")


if __name__ == "__main__":
    main()
