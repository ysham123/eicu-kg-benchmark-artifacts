# Public artifact policy

The public release is built from an allowlist. Its purpose is to expose the
benchmark contract, implementation logic, synthetic checks, and aggregate
paper results without republishing restricted source or derived data.

## Public

- source code and concise docstrings;
- machine-readable protocol constants;
- synthetic tests;
- aggregate metrics already reported in the paper;
- release-file checksums;
- schemas and instructions for credentialed regeneration.

## Not public

- raw eICU or patient-derived rows;
- raw medication strings or generated drug mappings;
- the generated candidate list or its data-derived ordering;
- patient and visit identifiers, split assignments, targets, or HARD masks;
- graph tables, diagnosis signatures, cached scores, or predictions;
- model parameters, gate bundles, or checkpoints;
- exploratory notebooks containing local paths or competing protocol variants.

Aggregate counts and metrics are not sufficient to reconstruct a patient
record and are included only when they are part of the reviewed paper claim.

## Code comments and docstrings

Docstrings are retained. They should state inputs, outputs, invariants, and
failure conditions. Conversational notes, private paths, repeated experiment
banners, and claims such as "official" or "final" are not part of the public
implementation.

## Before a release

Run:

```bash
pytest -q
python scripts/verify_release.py
sha256sum -c MANIFEST.sha256
```

The release is not ready if any command fails.
