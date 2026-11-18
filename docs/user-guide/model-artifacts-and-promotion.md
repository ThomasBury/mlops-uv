---
title: Model Artifacts And Promotion
description: Inspect AceBet schema-v2 bundles, checksum validation, and explicit local promotion.
---

# Model Artifacts And Promotion

AceBet exports a model only after comparing a constant baseline, the prepared
Elo probability, and LightGBM on the same date-grouped temporal selection holdout. The
constant candidate is comparison-only and is never serialized. No independent
final test is implemented; the later partition is used for model selection.

## Selection policy

Log loss is the only selection metric. Accuracy, Brier score, and ROC AUC remain
visible diagnostics but cannot override it.

1. Elo or LightGBM is servable when its unrounded log loss improves on constant
   by at least `0.005`.
2. If neither qualifies, training records `no_candidate_beats_constant`, raises,
   and writes no bundle.
3. If only one qualifies, that candidate is selected.
4. If both qualify, LightGBM replaces Elo only when it improves on Elo by at
   least `0.005`; otherwise Elo is preferred.

Equality at either boundary passes. The canonical window currently selects Elo:

| Candidate | Log loss |
|---|---:|
| Constant | 0.6931 |
| Elo | 0.6021 |
| LightGBM | 0.6181 |

## Bundle schema v2

An export consists of exactly three same-stem files:

```text
model_YYYY-MM-DD-HH-MM-SS.joblib
model_YYYY-MM-DD-HH-MM-SS.metadata.json
model_YYYY-MM-DD-HH-MM-SS.modelcard.md
```

The metadata records:

- the project-relative dataset path and complete Feather-file SHA-256
- requested and observed windows, cutoff, training/selection holdout date ranges (`test` in metadata), and row counts
- the complete ordered numeric and categorical feature contract
- all four metrics for constant, Elo, and LightGBM
- training prevalence, improvements over constant, servability, and LightGBM parameters
- both selection margins, computed deltas, selected kind, and reason
- the model filename, kind, exact `acebet` version, serialization marker, and SHA-256
- optional MLflow run ID and export timestamp

The model card presents the same candidate comparison and identity for human
review, including the integrity warning below.

## Canonical packaged bundle

The packaged promotion record selects `elo_baseline` with reason
`elo_preferred_challenger_margin_not_met`. These are the filenames and full
SHA-256 values recorded for the packaged example.

| File | SHA-256 |
| --- | --- |
| `model_2026-08-02-12-59-07.joblib` | `a51b64ddddc094260ff41aa68beba9662bdb7769076130f55b1ed54a51a703c7` |
| `model_2026-08-02-12-59-07.metadata.json` | `62cd4743f050eae1acc0b156307ce16b43f5b1b4d0e23cc7f20825120884afd0` |
| `model_2026-08-02-12-59-07.modelcard.md` | `a1b42707746c9c21ea2394b784c165fa2ecccc2c433328c534e34b161b2f0c43` |
| `data/atp_data_production.feather` | `cb27873cafc691f2e0e89a9e44478da240bd6ed3f507cb283b1ea2e720c044d3` |

## Promotion

```bash title="Export and promote a local bundle"
just rebuild-artifacts models
just serve-with-assets
```

Promotion writes `models/promoted_model.json`. Schema v2 binds the three
filenames to the selected model kind, exact package version, model SHA-256,
metadata SHA-256, model-card SHA-256, and selection reason. Promotion recomputes
selection from candidate metrics and validates every identity without
unpickling the model.

`serve-with-assets` sets `ACEBET_MODEL_DIR` to `models/` and
`ACEBET_DATA_DIR` to `data/`, so this promoted pointer and the refreshed player
state become the production serving inputs instead of merely being local build
outputs.

Schema-v1 metadata and pointers are deliberately unsupported. Regenerate and
promote them with the current workflow.

## Loading and trust boundary

Directory loading requires `promoted_model.json`; there is no newest-filename
fallback. A direct joblib path remains available to trusted tooling, but its
complete v2 sidecars are still mandatory.

Before `joblib.load`, the loader validates the pointer hashes, metadata schema,
feature contract, recomputed selection, selected kind, model digest, and exact
installed `acebet` version. After deserialization it verifies the Python type
and fitted feature contract for Elo or LightGBM.

!!! warning
    Checksums provide integrity, not authenticity. Joblib relies on pickle and
    can execute code while loading. Only load AceBet bundles obtained from a
    trusted source; a matching checksum does not make an untrusted artifact safe.

## MLflow

With `just train-local-mlflow`, AceBet logs namespaced metrics for every
candidate, selection margins and deltas, selected kind/reason, and the selected
sklearn model with a signature and input example. When exporting, all three
bundle files are logged. A total selection failure logs its comparisons and
reason, but no selected model or bundle.

MLflow is experiment tracking here; the checksum-bound local promotion pointer
remains the serving source of truth.
