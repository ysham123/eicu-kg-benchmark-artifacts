# eICU-KG

eICU-KG is a leakage-audited benchmark for held-out medication-concept edge
ranking on a clinical knowledge graph derived from the eICU Collaborative
Research Database. The benchmark contribution is the evaluation contract:
one fixed candidate space, patient-disjoint splits, a training-defined
long-tail slice, and explicit separation of aggregate and tail performance.

This repository contains public code, configuration, synthetic tests, and
aggregate results. It contains no eICU rows or patient-derived artifacts.
Access to eICU-CRD v2.0 is controlled by PhysioNet; credentialed users must
regenerate restricted intermediate artifacts locally. See [DATA.md](DATA.md).

## Benchmark contract

- Task: rank the held-out `DRUG_OF_CONCEPT` tail for an observed visit context.
- Candidate space: 147 medication concepts, fixed for every method.
- Split unit: patient (`uniquepid`), with 356/54/47 target-bearing patients in
  train/validation/test.
- Test set: 4,939 target edges.
- HARD: 1,886 test queries whose true concept occurs at most 383 times in
  training. This is 38.19% of test and covers 137 of 147 concepts.
- Ranking convention: `1 + sum(scores > true_score)`. Ties with the true score
  therefore receive the same optimistic rank.
- Primary metrics: mean reciprocal rank (MRR) and Hits@10 on both full test and
  HARD.

HARD is a training-frequency slice. It is not a diagnosis-signature OOV,
zero-shot, temporal, or prospective prediction setting.

## Reference results

| Method | Test MRR | Test H@10 | HARD MRR | HARD H@10 |
|---|---:|---:|---:|---:|
| DXPOP-Signature | 0.422 | 0.617 | 0.034 | 0.024 |
| DXPOP-Additive | 0.425 | 0.620 | 0.037 | 0.030 |

DXPOP-Signature is the exact diagnosis-signature prior used as the parent of
the CSR-HARD case study. It produces 3,048/4,939 test hits and 46/1,886 HARD
hits. DXPOP-Additive is a separate per-diagnosis implementation used for the
bin-wise comparator; the two names and values are not interchangeable.

## Limited case study: CSR-HARD

Constrained Selective Reranking on HARD (CSR-HARD) keeps DXPOP-Signature as the
default. A learned gate selects 7% of HARD rows, and only the DXPOP top-25
candidates in those rows receive an RGCN score contribution. The fixed primary
protocol is `q=0.07`, `M=25`, and `gamma=1.0`.

Across seeds 0, 1, and 2, CSR-HARD produces 54, 63, and 59 HARD top-10 hits.
The mean is 58.67 hits, or H@10 `0.031106 +/- 0.002391`, versus 46 hits for
DXPOP-Signature. The increase is 12.67 hits (27.5%). Mean HARD MRR is
`0.034272 +/- 0.000164`. All three runs have zero lost@10, and head score rows
remain unchanged. CSR-HARD is a benchmark-side analysis, not a clinical
routing policy.

The validation rule is a two-sided aggregate-stability guard on the complete
validation split:

```text
abs(delta MRR) <= 0.001 and abs(delta Hits@10) <= 0.001
```

Temperature scaling at `T*=1.1` is a separate calibration diagnostic. It is
not used to build gate features or train the gate.

## Install and test

```bash
python -m pip install -e '.[test]'
pytest -q
python scripts/verify_release.py
```

These commands use synthetic fixtures and do not require eICU. The credentialed
reproduction path and expected private artifact contract are documented in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Repository layout

```text
protocol.yaml              machine-readable benchmark and CSR protocol
src/eicu_kg/               metrics, HARD, CSR, and leakage-audit code
loaders/                   credentialed eICU-to-Neo4j ingestion helpers
results/paper_metrics.csv  verified aggregate, non-patient results
tests/                     synthetic invariant tests
docs/                      data access, artifact policy, reproduction notes
```

## Public artifact boundary

The candidate list, drug-normalization mapping outputs, patient split
assignments, graph Parquets, HARD masks, cached scores, and model checkpoints
are not publicly redistributed. They must be regenerated inside a
credentialed environment. The release scanner fails if a restricted filename
or file type is tracked.

## License

The source code is released under the MIT License. eICU and external
terminologies remain subject to their own access terms and licenses.
