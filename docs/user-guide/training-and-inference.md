---
title: Training and Inference
description: Use this page for the practical local training, inference, and asset-resolution workflow that sits next to the main tutorial.
---

# Training and Inference

This page is a practical workflow guide, not part of the sequential tutorial.
Use it when you want the concrete local training, promotion, and inference path
in one place.

## Training Flow

```bash title="Run the non-mutating training checks"
just data-check
just feature-state-check
just train-check
```

That sequence validates the canonical prepared data, exercises player-state
generation in temporary storage, and trains and compares all candidates without exporting or promoting a model.

Training uses one deterministic date-grouped temporal selection holdout. Preparation
normalizes and stably sorts calendar dates, then retains them separately from
the feature matrix. The nominal cutoff is two-thirds through the selected rows
and moves to the beginning of that date: training receives strictly earlier
dates, while the selection holdout receives the cutoff date and everything later. Training
fails with a contextual error if the window is reversed, empty, too small,
non-monotonic, or cannot produce two non-empty partitions that each contain
both target classes. The standard recipes then score the fitted candidates on
the following year of matches. Existing Python `test` names and metadata fields
under `dataset` refer to the selection holdout.

Artifact changes are a separate, explicit workflow:

```bash title="Intentionally refresh, export, and promote artifacts"
just rebuild-artifacts
```

That command refreshes project and packaged player-state files, then exports and
promotes a model under ignored `var/models/`. Use `just rebuild-artifacts models`
only when you intentionally want the serving model under `models/`.

To prove the committed raw sample can be rebuilt without replacing canonical or
packaged data, run:

```bash title="Rebuild the raw sample in temporary storage"
just data-check-sample
```

For a caller-selected destination, the generic builder requires both paths:

```bash title="Explicit raw-to-prepared output"
just data-build data/raw/atp_matches_sample.csv build/atp_data_sample.feather
```

An exported bundle includes:

- `model_*.joblib`
- `model_*.metadata.json`
- `model_*.modelcard.md`

Optional MLflow tracking is available through:

```bash title="Train with MLflow enabled"
just train-local-mlflow
```

That recipe opts into the `mlops` dependency group before training. In the
default local setup, MLflow uses:

- `sqlite:///mlflow.db` as the tracking URI
- `acebet` as the experiment name
- `mlartifacts/` as the local artifact root

The training code computes accuracy, log loss, Brier score, and ROC AUC for the
constant, Elo, and LightGBM candidates from the same selection holdout. Only unrounded log
loss selects a model: learned candidates must beat constant by `0.005`, and
LightGBM must then beat a servable Elo candidate by `0.005`. These margins are
fixed in the selection policy. The standard Just recipes pass
`--post-selection-end-date 2018-03-04` to score the same candidates on dates
strictly after `--end-date`. The export records those results under
`post_selection` in metadata and in the model card. MLflow logs selection
metrics and deltas, and attaches the three bundle files when exported.

The training CLI uses the prepared dataset from a repository checkout by
default. When running from a wheel, pass `--data-path` with a prepared Feather
file. For export from a wheel, also choose a writable `--output-dir`.

For the exact local artifact bundle and promotion workflow, read
[Model Artifacts and Promotion](./model-artifacts-and-promotion.md).

## Feature Contract

Training and serving share one ordered allowlist:

| Position | Type | Column |
| ---: | --- | --- |
| 1 | numeric | `best of` |
| 2 | numeric | `rank_p1` |
| 3 | numeric | `rank_p2` |
| 4 | numeric | `elo_p1` |
| 5 | numeric | `elo_p2` |
| 6 | numeric | `proba_elo` |
| 7 | numeric | `year` |
| 8 | numeric | `month` |
| 9 | numeric | `day` |
| 10 | numeric | `rank_diff` |
| 11 | categorical | `p1` |
| 12 | categorical | `p2` |
| 13 | categorical | `series` |
| 14 | categorical | `court` |
| 15 | categorical | `surface` |
| 16 | categorical | `round` |
| 17 | categorical | `best_ranked` |

`select_model_features(...)` returns exactly this order and silently discards
every unrelated raw column. If any required column is absent, training or
prediction fails before fit/scoring with all missing names in the error. Feature
alignment never creates a missing column or pads it with `pd.NA`.

The preprocessing `ColumnTransformer` passes the 10 numeric values through
unchanged. It applies `OrdinalEncoder` only to the 7 categorical fields and
marks those codes as unordered categories for LightGBM's native categorical
splits. Unknown categories are accepted at prediction time and handled as
missing categories. Legacy artifacts that ordinal-encoded the entire dataframe,
or that expose a different feature order, fail during loading.

Derived and online behavior remains shared:

- The raw builder derives `proba_elo` from raw ELO columns.
- Online inference recomputes `proba_elo` from the latest player ELO values.
- Date derives `year`, `month`, and `day`, then remains outside the model.
- Request field `best_of` becomes the internal `"best of"` feature.
- Online scoring requires explicit `series`, `court`, `surface`, `round`,
  and `best_of`; it has no silent match-context defaults.
- Legacy `atp`, `location`, and `tournament` request fields are rejected.

## Inference Flow

Start the API:

```bash title="Start the API"
cp .env.example .env
just serve
```

Leave that command running. In another terminal, get a token:

```bash title="Request a bearer token"
just token-json
```

Exercise prediction in testing mode:

```bash title="Exercise testing-mode prediction"
just predict-testing
```

Exercise prediction in production mode:

```bash title="Exercise production-mode prediction"
just predict-production
```

Both prediction modes authenticate the same way. Runtime mode is selected by
route: `just predict-production` calls `/predict/production`, while
`just predict-testing` calls `/predict/testing`. The testing route is available
only when `ACEBET_ENV` is `development` or `test`.

## Asset Resolution

- With no overrides, production uses the packaged promoted bundle and packaged
  `latest_player_stats.feather` for zero-setup serving.
- `ACEBET_MODEL_DIR` overrides only the production model directory and requires
  `promoted_model.json` there.
- `ACEBET_DATA_DIR` overrides only the production data directory and requires
  `latest_player_stats.feather` there.
- Production startup checks the player-state SHA-256 recorded by training.
  To serve an older bundle with directory overrides, set
  `ACEBET_PLAYER_STATS_SHA256` to the digest of its trusted player-state file.
- Testing always uses packaged sample player state and the packaged promoted
  model, regardless of production overrides.

Every model directory must contain `promoted_model.json`; there is no
newest-filename fallback. A missing configured path fails startup rather than
silently using packaged assets. The loader validates schema-v2 sidecars, hashes,
selection, model kind, and feature contract before joblib deserialization.
The package version records which AceBet version exported the bundle.
Direct-file loading is reserved for trusted tooling and still requires both
complete sidecars.

Set `ACEBET_MAX_SNAPSHOT_AGE_DAYS` to reject snapshots older than your limit.
With the setting unset, successful requests using a snapshot older than 60
days emit a warning with both ages.

To serve the artifacts produced by local promotion:

```bash
just rebuild-artifacts models
just serve-with-assets
```

Checksums detect accidental or malicious changes but do not establish who
created an artifact. Because joblib uses pickle, load bundles only from trusted
sources.

## Manual HTTP Example

```bash title="Token request with curl"
curl -X POST http://127.0.0.1:8000/token \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=johndoe&password=secret'
```

Those are the default demo credentials. Override them with
`ACEBET_DEMO_USERNAME` and `ACEBET_DEMO_PASSWORD` if needed.

```json title="Example prediction payload"
{
  "p1_name": "Fognini F.",
  "p2_name": "Jarry N.",
  "date": "2018-03-04",
  "series": "ATP250",
  "court": "Indoor",
  "surface": "Clay",
  "round": "The Final",
  "best_of": 3
}
```

Send that body to `/predict/production` for production assets or to
`/predict/testing` for packaged sample/testing assets outside production.
The payload uses the indoor clay context of the 2018 Brasil Open final, just
as `just predict-production` does. The packaged snapshots come from the same
match row, so this is a replay of a known result.
