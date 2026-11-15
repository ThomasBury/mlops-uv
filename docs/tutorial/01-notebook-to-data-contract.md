---
title: Tutorial 1 - The Prepared-Data Boundary
description: Start from the committed prepared dataset, understand the raw sample's limits, and make upstream lineage gaps explicit.
---

# Tutorial 1: The Prepared-Data Boundary

AceBet's executable training story begins at
`data/atp_data_production.feather`: a committed, prepared dataset containing
**44,617 rows**. `prepare_data(...)` validates that boundary, normalizes dates,
stably orders equal-date rows, and deterministically re-derives calendar and
ranking features.

This is a prepared-data contract, not a claim that the whole upstream data
pipeline can be reproduced from this repository.

## Why This Exists

A **prepared dataset** is data already cleaned into the shape training expects.
This boundary stops a convenient sample file from being mistaken for the
historical data behind a model decision. The checks below prove the committed
input is structurally usable; they cannot prove its upstream history.

## Run The Non-Mutating Checks

```bash
uv sync --locked
just data-check
just data-check-sample
just feature-state-check
```

`data-check` validates the committed 44,617-row Feather file.
`data-check-sample` rebuilds the small raw sample under temporary storage.
`feature-state-check` derives player snapshots only in a temporary directory.
None of these commands replaces a tracked data or model artifact.

The companion notebook is also executable without changing its committed cells,
outputs, or execution counts:

```bash
just notebook-check
```

## What The 100-Row Sample Proves

`data/raw/atp_matches_sample.csv` contains 100 rows. It exists only to
demonstrate CSV normalization and deterministic derived-feature mechanics such
as `proba_elo`, calendar fields, `rank_diff`, and `best_ranked`.

It **cannot reproduce** the 44,617-row production Feather file. A caller-selected
raw rebuild must therefore target ignored or temporary storage:

```bash
just data-build data/raw/atp_matches_sample.csv var/tutorial/atp_data_sample.feather
```

Never use the canonical production Feather path as the output of that tutorial
example.

## Historical Data And Runtime State

Training consumes historical match rows. Serving instead needs the latest known
state for each requested player: `player`, `rank`, `elo`, and
`as_of_date`. `build_latest_player_stats(...)` derives this compact lookup
table, and the API reports the exact snapshot date and age it used.

That separation is useful, but its provenance has limits:

- the original upstream source and its licensing terms are not recorded;
- Elo histories are supplied in the prepared inputs rather than generated from
  an auditable match-history process;
- point-in-time lineage connecting each training row and serving snapshot to an
  upstream snapshot is absent.

These gaps prevent claims of complete data reproducibility. They also prevent
independent proof that every ranking, Elo value, and snapshot was causally
available at its match time. The code enforces a clear downstream boundary; it
does not manufacture missing upstream evidence.

!!! info "Shortcut"

    **Why it is enough here:** schema validation and deterministic rebuilding
    make the downstream input inspectable. **Add it when…** source ownership,
    licensing, Elo generation, or point-in-time lineage must support stronger
    causal or production claims. See [Tutorial Boundaries](../tutorial-shortcuts-vs-production.md).

## Experiment: Remove A Required Field

**Prerequisites:** run `uv sync --locked` and work from the repository root.
**Prediction:** removing `rank_p1` from a temporary sample names that missing
field and produces no prepared output.

```bash
uv run python - <<'PYCODE'
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from acebet.dataprep.raw_to_prepared import build_prepared_dataset

with TemporaryDirectory() as directory:
    root = Path(directory)
    sample = pd.read_csv("data/raw/atp_matches_sample.csv")
    sample.drop(columns="rank_p1").to_csv(root / "missing.csv", index=False)
    output = root / "prepared.feather"
    try:
        build_prepared_dataset(root / "missing.csv", output)
    except ValueError as error:
        print(error)
    else:
        raise AssertionError("The missing field should have been rejected")
    assert not output.exists()
    print("No prepared output.")
PYCODE
```

Expected output:

```text
Raw dataset is missing required columns: ['rank_p1']
No prepared output.
```

**Explain:** why must this fail before any derived features or output writes?
**Cleanup:** the temporary directory is removed automatically; the sample is
unchanged.

Continue with
[Model Validity, Selection, And The Feature Contract](./02-feature-contract-training-and-export.md).
