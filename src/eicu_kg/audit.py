"""Leakage and public-release checks that do not require eICU data."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

RESTRICTED_SUFFIXES = {
    ".ckpt",
    ".db",
    ".h5",
    ".hdf5",
    ".ipynb",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".sqlite",
    ".tsv",
}
RESTRICTED_PARTS = {
    "artifacts",
    "checkpoints",
    "data",
    "logits",
    "mapping",
    "splits",
}
RESTRICTED_NAMES = {
    "candidate_concepts_k147.csv",
    "drug_mapping_v1.5_full.csv",
}


def count_all_zero_rows(scores: ArrayLike) -> int:
    """Count score rows that are exactly zero in every candidate coordinate."""

    array = np.asarray(scores)
    if array.ndim != 2:
        raise ValueError("scores must have shape [n_queries, n_candidates]")
    return int(np.sum(np.all(array == 0, axis=1)))


def assert_patient_disjoint(
    train_ids: Iterable[object],
    validation_ids: Iterable[object],
    test_ids: Iterable[object],
) -> None:
    """Raise if a patient identifier occurs in more than one split."""

    train = set(train_ids)
    validation = set(validation_ids)
    test = set(test_ids)
    overlaps = {
        "train/validation": train & validation,
        "train/test": train & test,
        "validation/test": validation & test,
    }
    nonempty = {name: values for name, values in overlaps.items() if values}
    if nonempty:
        summary = ", ".join(
            f"{name}={len(values)}" for name, values in nonempty.items()
        )
        raise ValueError(f"patient split overlap detected: {summary}")


def assert_targets_withheld(
    training_edges: Iterable[tuple[object, object, object]],
    validation_targets: Iterable[tuple[object, object, object]],
    test_targets: Iterable[tuple[object, object, object]],
) -> None:
    """Raise if a validation or test target edge appears in the training graph."""

    training = set(training_edges)
    overlap_validation = training & set(validation_targets)
    overlap_test = training & set(test_targets)
    if overlap_validation or overlap_test:
        raise ValueError(
            "target leakage detected: "
            f"validation={len(overlap_validation)}, test={len(overlap_test)}"
        )


def restricted_release_paths(root: str | Path) -> list[str]:
    """Return tracked-like paths that violate the public artifact boundary."""

    root_path = Path(root).resolve()
    violations: list[str] = []
    for path in root_path.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root_path)
        lowered_parts = {part.lower() for part in relative.parts[:-1]}
        lower_name = relative.name.lower()
        is_named_restricted = lower_name in RESTRICTED_NAMES
        has_restricted_suffix = relative.suffix.lower() in RESTRICTED_SUFFIXES
        is_in_restricted_directory = bool(lowered_parts & RESTRICTED_PARTS)
        is_reviewed_result = str(relative) == "results/paper_metrics.csv"
        if (
            is_named_restricted
            or has_restricted_suffix
            or (is_in_restricted_directory and not is_reviewed_result)
        ):
            violations.append(str(relative))
    return sorted(set(violations))
