---
title: Tutorial 2 - Model Validity And Selection
description: Evaluate prediction-time availability, temporal validity, typed preprocessing, baselines, probability metrics, and the canonical Elo selection.
---

# Tutorial 2: Model Validity, Selection, And The Feature Contract

A repeatable training command is not enough. Before exporting anything, ask
whether the inputs could exist at prediction time, whether evaluation respects
time, and whether the challenger improves the probability forecast that will
actually be served.

## Why This Exists

A **feature** is one input value the model uses. **Leakage** is an input that
would not have been known when the prediction was needed. A **selection holdout** is the later partition used to compare candidates;
training uses earlier dates. A **baseline** is the
simple comparison a learned model must beat. The smallest comparison below
checks all four ideas before any model is exported.

## Run The Comparison Without Exporting

```bash
just train-check
just notebook-check
```

Both commands compare candidates in memory. They do not write or promote model
artifacts.

## Prediction-Time Availability Comes First

The shared allowlist contains 17 ordered features. Ten are numeric:
`best of`, `rank_p1`, `rank_p2`, `elo_p1`, `elo_p2`, `proba_elo`,
`year`, `month`, `day`, and `rank_diff`. Seven are categorical:
`p1`, `p2`, `series`, `court`, `surface`, `round`, and
`best_ranked`.

Labels, final scores, bookmaker fields, comments, and the raw date cannot reach
the estimator. The date is retained separately for splitting and derives only
calendar fields. Online scoring rebuilds the same contract from the request and
two player snapshots.

This allowlist reduces leakage risk, but it cannot prove causal availability of
the supplied Elo and snapshot histories. Chapter 1's missing point-in-time
lineage remains a real validity limitation.

## One Strict Date-Grouped Selection Holdout

The canonical requested window is `2015-03-04` through `2017-03-04`.
Preparation observes 5,236 rows from `2015-03-12` through `2017-03-04`.
The nominal two-thirds boundary moves to the first row on its calendar date:

| Partition | Rows | Date range |
| --- | ---: | --- |
| Train | 3,482 | 2015-03-12 through 2016-06-19 |
| Selection holdout | 1,754 | 2016-06-20 through 2017-03-04 |

The invariant is strict:
`max(train_date) < min(test_date)`. No calendar date appears on both sides.
Both partitions must be non-empty and contain both target classes. The later
partition selects the winner, so its metrics are selection evidence. No
independent final test is implemented. This is one selection holdout, not
cross-validation or evidence across multiple eras.

Python names such as `test_indices`, `X_test`, and `y_test`, and metadata
`test` fields are retained for compatibility. They all mean the selection
holdout, including the companion notebook's outputs.

## Preprocessing Respects Feature Type

The LightGBM challenger uses a `ColumnTransformer`:

- numeric columns pass through unchanged;
- only categorical columns receive `OrdinalEncoder` codes;
- those codes become unordered pandas categories for LightGBM's native
  categorical handling;
- unknown categories map to the configured unknown code.

The constant and Elo baselines use the same rows. Constant predicts the training
prevalence for every selection holdout row. Elo serves the prepared `proba_elo`. LightGBM
is the learned challenger, not the assumed winner.

## Canonical Candidate Evidence

All candidates are evaluated with one shared probability evaluator. Displayed
metrics use four decimal places; metadata retains their exact values:

| Candidate | Accuracy | Log loss | Brier score | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Constant | 0.5006 | 0.6931 | 0.2500 | 0.5000 |
| Elo | 0.6762 | 0.6021 | 0.2066 | 0.7410 |
| LightGBM | 0.6631 | 0.6181 | 0.2138 | 0.7239 |

Log loss is the only selection metric. A learned candidate must beat constant
by at least `0.005`. If both are servable, LightGBM replaces Elo only when it
also beats Elo by at least `0.005`. Comparisons use unrounded values.

The canonical result is therefore:

- selected kind: `elo_baseline`
- reason: `elo_preferred_challenger_margin_not_met`
- explanation: both learned candidates beat constant, but LightGBM does not
  beat Elo by the challenger margin.

## Probability Quality And Calibration Limits

Log loss rewards probability assigned to the observed class and heavily
penalizes confident mistakes. Brier score is also sensitive to probability
quality and is useful as a calibration-sensitive diagnostic. ROC AUC measures
ranking, while accuracy applies the repository's fixed decision threshold.

None of those diagnostics proves calibration. This repository has no reliability
gate, no post-hoc calibrator, and no acceptance policy based on a calibration
curve. The notebook may display a small reliability table for intuition, but
its 100-row sample is not calibration evidence.

!!! info "Shortcut"

    **Why it is enough here:** one selection holdout and probability diagnostics expose
    the selection decision without claiming calibration. **Add it when…**
    decisions need evidence across eras or calibrated probabilities; then add
    backtests, reliability gates, and a calibrator. See
    [Tutorial Boundaries](../tutorial-shortcuts-vs-production.md).

## Experiment: Change Hypothetical Candidate Evidence

**Prerequisites:** run `uv sync --locked` from the repository root.
**Prediction:** lowering only a copied LightGBM log loss below Elo by `0.006`
changes the selection to LightGBM. This is a **hypothetical policy experiment**,
not a new measured result or a model-quality claim.

```bash
uv run python - <<'PYCODE'
from copy import deepcopy
from pathlib import Path
from acebet.model_artifacts import compute_selection, read_json

metadata = read_json(Path("src/acebet/data/model_2026-08-02-12-59-07.metadata.json"))
candidates = deepcopy(metadata["candidates"])
print("Recorded:", compute_selection(candidates)["selected_model_kind"])
candidates["lightgbm"]["metrics"]["log_loss"] = candidates["elo"]["metrics"]["log_loss"] - 0.006
selection = compute_selection(candidates)
print("Hypothetical:", selection["selected_model_kind"], selection["reason_code"])
assert candidates != metadata["candidates"]
assert selection["reason_code"] == "lightgbm_beats_elo_margin"
PYCODE
```

Expected output:

```text
Recorded: elo_baseline
Hypothetical: lightgbm lightgbm_beats_elo_margin
```

**Explain:** why would rounded metrics be insufficient for a decision exactly
at the policy margin? Why does this experiment provide no independent final
test evidence?
**Cleanup:** all changes exist only in memory; no bundle is exported or promoted.

Continue with
[Artifact Reproducibility And Trust](./03-reproducibility-versioning-and-dependencies.md).
