---
title: Choose the bundle the service may load
description: Promote your training export and see how startup detects a changed file.
---

# 4. Choose the bundle the service may load { #tutorial-3-artifact-reproducibility-and-trust }

<a id="why-this-exists"></a>
<a id="inspect-without-regenerating"></a>
<a id="the-schema-v2-bundle"></a>

Training saved three files with the same base name. The `.joblib` holds the
saved estimator, the part that makes predictions. The `.metadata.json` holds
the evaluation and identity, and the `.modelcard.md` explains the result.
Together they form the **bundle**. Now designate that bundle for serving.

## Promote the export { #promotion-is-checksum-bound }

```bash
just promote-local var/tutorial/models
cat var/tutorial/models/promoted_model.json
```

**Promotion** chooses a bundle the service may load. The recipe selects the
only export in your tutorial directory and writes `promoted_model.json`.
Find `model_filename` and `model_sha256` in the record. SHA-256 is the
algorithm used to compute a checksum from a file's bytes. The record names
and binds all three files by checksum.

## Change one file in a copy { #experiment-check-hashes-then-alter-a-copy }

This example fully works through a diagnosis. The question is whether a
bundle can still be loaded after someone changes its model card.

```bash
uv run python - <<'PY'
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

from acebet.model_artifacts import resolve_model_artifact_path

with TemporaryDirectory() as directory:
    bundle = Path(directory) / "models"
    copytree("var/tutorial/models", bundle)
    model = resolve_model_artifact_path(bundle)
    print("Verified:", model.name)

    card = model.with_suffix(".modelcard.md")
    with card.open("a") as file:
        file.write("\nChanged in the temporary copy.\n")

    try:
        resolve_model_artifact_path(bundle)
    except ValueError as error:
        print(error)
    else:
        raise AssertionError("The changed card should have been rejected.")
PY
```

The first check resolves the intended model. The second reports
`Promotion record modelcard_sha256 mismatch`. It checks the card against
the recorded fingerprint and rejects the changed copy before loading the
model. The temporary copy is removed when the check finishes.

## Decide whether to trust the source { #integrity-is-not-authenticity }

A checksum can establish that bytes match the record. You still need to trust
who supplied the bundle and the record. Loading a joblib file can execute
code, so use an export you created or obtained from a trusted publisher.

<a id="regeneration-is-a-separate-companion-workflow"></a>

**Finish when** you can name the file the service is allowed to load, explain
what the edited copy demonstrated, and say why trust still matters. Keep the
original bundle and promotion record for
[lesson 5](./04-serving-async-and-monitoring.md). The
[artifact guide](../user-guide/model-artifacts-and-promotion.md) covers the
full record and the maintenance workflow.
