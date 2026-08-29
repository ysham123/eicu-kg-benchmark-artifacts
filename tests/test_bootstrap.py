import numpy as np

from eicu_kg.bootstrap import patient_cluster_bootstrap_difference


def test_patient_bootstrap_keeps_clustered_queries_together() -> None:
    parent = np.array([0, 0, 1, 1], dtype=bool)
    candidate = np.array([[1, 1, 1, 1]], dtype=bool)
    patients = np.array(["a", "a", "b", "b"])
    result = patient_cluster_bootstrap_difference(
        parent, candidate, patients, n_replicates=100, seed=7
    )
    assert result.point_difference == 0.5
    assert result.n_patients == 2
    assert set(result.replicate_differences).issubset({0.0, 0.5, 1.0})
