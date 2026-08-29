# Reproduction contract

This document separates checks that run on public synthetic data from stages
that must run in a credentialed environment.

## Public verification

From a clean checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
python scripts/verify_release.py
sha256sum -c MANIFEST.sha256
```

The tests verify the ranking convention, HARD construction, patient
disjointness, target withholding, gate feature order, constant eighth feature,
top-25 fusion boundary, fixed-budget rounding, and two-sided validation guard.
They do not use synthetic values as substitutes for paper results.

## Credentialed construction path

1. Obtain eICU-CRD v2.0 directly from PhysioNet and obtain the external
   terminology resources under their own licenses.
2. Load the required source columns in the credentialed environment. The
   helpers in `loaders/eicu_loaders.py` construct source nodes and relations in
   bounded batches.
3. Run the frozen medication-normalization stage locally. Its generated raw
   medication vocabulary, mapping table, and 147-item candidate list stay in
   the credentialed artifact root and are not committed.
4. Construct the four core modeling relations and the separate `HAS_VISIT`
   relation. Split by `uniquepid` with seed 42. Candidate construction,
   diagnosis-signature vocabulary, and the HARD boundary use TRAIN only.
5. Withhold validation and test `DRUG_OF_CONCEPT` targets while preserving
   transductive context. Verify pairwise patient disjointness and zero overlap
   between withheld targets and training targets.
6. Train the declared baselines. Store every score matrix with the same local
   candidate ordering and a private semantic hash.
7. Evaluate DXPOP-Signature and DXPOP-Additive under distinct identifiers.
   Build CSR gate labels from TRAIN only, train three gate initializations, and
   select the largest fixed budget accepted for all seeds using validation only.
8. Evaluate the frozen test protocol once and compare the resulting aggregate
   values with `results/paper_metrics.csv`.

The frozen CSR implementation selects shortlist membership with NumPy
`argpartition`. A private run manifest must therefore record both the NumPy
version and the candidate-order semantic hash. This preserves the measured
implementation without silently changing tie-boundary behavior before the
paper values are recomputed.

## Required local checks

A private run manifest should record at least:

- source and configuration versions;
- candidate semantic hash and `K=147`;
- array names, shapes, dtypes, and SHA-256 checksums;
- patient and target counts per split;
- zero patient overlap and zero target-edge leakage;
- HARD cutoff, query count, and concept count;
- baseline implementation identifier;
- CSR seed, gate feature version, and fixed inference constants.
- the number of all-zero rows in every score matrix;
- the NumPy version and top-25 shortlist-membership hash.

Evaluation must stop on a missing file, mismatched shape, changed candidate
ordering, target index outside `[0,147)`, or failed leakage audit. It must not
search for the newest matching file or silently substitute DXPOP-Additive for
DXPOP-Signature.

The frozen candidate ordering includes source identifier `GTC:0`. It is
retained to preserve the evaluated benchmark and must be documented as a
dataset-specific source category, not interpreted as a validated therapeutic
class. Removing or relabeling it would define a new benchmark and requires a
full recomputation.

## Interpretation boundary

The task is held-out concept-edge ranking, not next-medication forecasting,
treatment recommendation, or clinical deployment. CSR-HARD is included only
as a constrained benchmark-side case study.
