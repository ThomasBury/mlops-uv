---
title: Tutorial 3 - Artifact Reproducibility And Trust
description: Understand schema-v2 metadata, checksum-bound promotion, locked dependencies, and the trusted-only joblib boundary.
---

# Tutorial 3: Artifact Reproducibility And Trust

Model reproducibility requires more than keeping a joblib file. AceBet binds the
selected estimator to evaluation evidence, dataset identity, feature order, and
an explicit promotion pointer.

## Why This Exists

An **artifact** is a saved model plus the information needed to identify and
check it. **Promotion** is the explicit act of choosing which saved bundle a
service may load. The commands below re-run the comparison without changing
that choice.

## Inspect Without Regenerating

```bash
uv sync --locked
uv run python -m acebet.train.train --help
just train-check
```

`--locked` requires `uv.lock` to agree with `pyproject.toml`; it does not
silently refresh the lockfile. The development group includes the test, docs,
type-check, and notebook tools. Runtime images exclude that group.

## The Schema-V2 Bundle

The committed promoted bundle is:

- `model_2026-08-02-12-59-07.joblib`
- `model_2026-08-02-12-59-07.metadata.json`
- `model_2026-08-02-12-59-07.modelcard.md`

Metadata records the prepared-dataset path and SHA-256, requested and observed
windows, cutoff and partition ranges, row counts, exact ordered feature
contract, every candidate metric, selection margins and deltas, selected kind
and reason, package version, serialization type, and model checksum.

The model SHA-256 is
`a51b64ddddc094260ff41aa68beba9662bdb7769076130f55b1ed54a51a703c7`.
The metadata SHA-256 is
`5e18e66678e8bf766b38a13ed8dcbd13d784c0e33ad4cf827a472399a450115a`.
The model-card SHA-256 is
`e9b9d1a5db70211604494ca586ab7ff402e94163e54aaf1a33a4819ce9811baf`.
The prepared dataset SHA-256 is
`cb27873cafc691f2e0e89a9e44478da240bd6ed3f507cb283b1ea2e720c044d3`.

The package identity is `acebet==0.2.1`. Release preparation updates the bundle
version metadata and checksums; the estimator bytes, candidate metrics, and
selected model are unchanged.

## Promotion Is Checksum-Bound

`promoted_model.json` names all three files and binds their hashes, the model
kind `elo_baseline`, package version, and selection reason
`elo_preferred_challenger_margin_not_met`.

Directory loading requires that pointer. There is **no newest-file fallback**.
Before deserialization, the loader verifies pointer hashes, schema version,
dataset/model identity, feature contract, package version, and selection
recomputed from the candidate metrics. After loading it verifies estimator type
and fitted contract.

## Integrity Is Not Authenticity

A matching checksum detects changed bytes. It does not prove who produced,
reviewed, or authorized those bytes. Joblib uses pickle semantics and may
execute code while loading, so this is a trusted-only boundary. Never treat a
hash match as permission to load an artifact from an untrusted source.

The bundle makes the downstream artifact reproducible and inspectable. It does
not repair Chapter 1's missing upstream source, license, Elo-generation, or
point-in-time lineage.

!!! info "Shortcut"

    **Why it is enough here:** checksums and a required local pointer prevent
    accidental bundle drift. **Add it when…** another person or environment
    must authorize artifacts; then add signing, a registry, and approval
    workflow. See [Tutorial Boundaries](../tutorial-shortcuts-vs-production.md).

## Regeneration Is A Separate Companion Workflow

!!! warning "Intentionally mutating"

    `just rebuild-artifacts` refreshes player-state files, exports a bundle,
    and promotes it. It is a companion maintenance workflow, not a sequential
    tutorial check. Use the non-exporting commands above unless you explicitly
    intend to replace artifacts.

## Experiment: Check Hashes, Then Alter A Copy

**Prerequisites:** run `uv sync --locked` from the repository root. Use only the
trusted bundled files supplied with this checkout.
**Prediction:** all three pointer hashes match at first; altering the copied
model card then causes checksum rejection before deserialization.

```bash
uv run python - <<'PYCODE'
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from acebet.model_artifacts import read_json, resolve_model_artifact_path, sha256_file

source = Path("src/acebet/data")
record = read_json(source / "promoted_model.json")
with TemporaryDirectory() as directory:
    copied = Path(directory)
    copyfile(source / "promoted_model.json", copied / "promoted_model.json")
    for kind in ("model", "metadata", "modelcard"):
        name = record[f"{kind}_filename"]
        assert sha256_file(source / name) == record[f"{kind}_sha256"]
        copyfile(source / name, copied / name)
    resolve_model_artifact_path(copied)
    print("All three pointer hashes match.")
    card = copied / record["modelcard_filename"]
    card.write_bytes(card.read_bytes() + b"\nChanged only in the temporary copy.\n")
    try:
        resolve_model_artifact_path(copied)
    except ValueError as error:
        print(error)
    else:
        raise AssertionError("The altered bundle should have been rejected")
PYCODE
```

Expected checkpoint: `All three pointer hashes match.`, followed by a
`Promotion record modelcard_sha256 mismatch` error. This rejects the altered
copy without loading the estimator.

**Explain:** why does a matching checksum prove integrity but not authorize
loading an untrusted joblib file?
**Cleanup:** all copied files are removed automatically; the promotion pointer
and original bundle remain unchanged.

Continue with
[Serving Validated Predictions](./04-serving-async-and-monitoring.md).
