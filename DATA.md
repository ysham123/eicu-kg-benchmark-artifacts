# Data access and redistribution boundary

This repository contains no clinical rows and no patient-derived benchmark
artifacts. It is not a redistribution of eICU-CRD.

## Primary source

The benchmark is derived from eICU Collaborative Research Database v2.0:

https://physionet.org/content/eicu-crd/

PhysioNet controls access. A reproducer needs an approved PhysioNet account,
the required training, and acceptance of the applicable Data Use Agreement.
The source tables are obtained directly from PhysioNet, never from this
repository.

## External terminology sources

The credentialed construction path also uses locally licensed terminology
resources, including UMLS Metathesaurus and SNOMED CT, and the open PrimeKG
resource. Users are responsible for obtaining each source under its own terms.
No terminology credential or API key is committed.

## Files that are intentionally absent

The public release does not include:

- raw or processed eICU tables;
- the 147-item candidate list;
- drug-normalization mapping outputs;
- patient, stay, visit, or target identifiers;
- split assignments or manifests containing patient identifiers;
- graph tables, targets, HARD masks, diagnosis signatures, or visit maps;
- cached DXPOP-Signature or RGCN scores;
- predictions, checkpoints, or trained models.

These outputs are generated and checked inside the credentialed environment.
They are not described as available on request.

## Local artifact contract

Evaluation code expects locally regenerated arrays with a common candidate
ordering and a private manifest that records shapes and semantic hashes. At a
minimum, a private evaluation bundle contains labels, HARD masks, and the
DXPOP-Signature and RGCN score matrices for train, validation, and test.
Before evaluation, the reproducer must verify candidate ordering, array
lengths, target-index bounds, split disjointness, and target-edge withholding.

## Commit protection

`.gitignore` blocks common clinical, model, and serialized artifact types.
`scripts/verify_release.py` provides a second fail-closed scan of tracked
release paths. These controls reduce accidental commits but do not replace the
terms of the source licenses or Data Use Agreement.
