# Packaged model and data record

The authenticated predictor replays the 2018 Fognini–Jarry final from **prepared Feather data**. The production dataset has 44,617 rows. The original production source and complete transformation history are unavailable. A separate 100-row raw sample lets you inspect how match context and player state are derived. [Download the starter notebook](../assets/notebooks/01-starter-feature-and-training-flow.ipynb) · [Run the packaged predictor](../getting-started.md)

Run `just data-check-sample` to rebuild the small sample into temporary storage and `just notebook-check` to execute the notebook without overwriting it. `just train-check` compares the constant, Elo and LightGBM candidates on prepared data. The standard recipes also score them on a later period. A wheel user supplies a prepared Feather file with `--data-path` and a writable `--output-dir` for export.

## Which matches informed selection?

| Use | Rows | Date range |
| --- | ---: | --- |
| Fit candidates | 3,482 | 2015-03-12 through 2016-06-19 |
| Compare candidates | 1,754 | 2016-06-20 through 2017-03-04 |
| Evaluate after selection | 2,630 | 2017-03-05 through 2018-03-04 |

Keeping an entire date on one side of the split makes the fitting period
strictly earlier. After selection, training scores the same fitted candidates
on the third period. Elo log loss rises from 0.6021 on the selection holdout
to 0.6295 in that later window. Find both results in the model card and the
later metrics under `post_selection` in the metadata.


The cutoff moves to the beginning of its calendar date. The historical LightGBM tuning window was unrecorded, so earlier exposure to the later evaluation period is unknown.

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

| Candidate | Accuracy | Log loss | Brier score | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Constant | 0.5006 | 0.6931 | 0.2500 | 0.5000 |
| Elo | 0.6762 | 0.6021 | 0.2066 | 0.7410 |
| LightGBM | 0.6631 | 0.6181 | 0.2138 | 0.7239 |

## Bundle schema v2

An export consists of exactly three same-stem files:

```text
model_<timestamp>_<unique-id>.joblib
model_<timestamp>_<unique-id>.metadata.json
model_<timestamp>_<unique-id>.modelcard.md
```

The metadata records:

- the dataset path (filename for files outside the checkout) and complete Feather-file SHA-256
- requested and observed windows, cutoff, training/selection holdout date ranges (`test` in metadata), and row counts
- the complete ordered numeric and categorical feature contract
- all four metrics for constant, Elo, and LightGBM
- training prevalence, improvements over constant, servability, and LightGBM parameters
- both selection margins, computed deltas, selected kind, and reason
- the post-selection date range, row count, and metrics when requested
- the SHA-256 of the production player state derived from the same dataset
- the model filename, kind, training package version, serialization marker, and SHA-256
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
| `model_2026-08-02-12-59-07.metadata.json` | `b5f0b8bf32b98e40bd7da38f9244cf194c8a015e5a1e9a1217db33558f2573e5` |
| `model_2026-08-02-12-59-07.modelcard.md` | `91393f6758f7090294674145f5db4efc268bcdf2ba0a9b531a0d178b84ff196a` |
| `data/atp_data_production.feather` | `cb27873cafc691f2e0e89a9e44478da240bd6ed3f507cb283b1ea2e720c044d3` |
| `latest_player_stats.feather` | `bad2d0f978840b0daadc4416cbfe49375e1abdf57a09b8edfc8a6b9f28f7bb8b` |

## Promotion

```bash title="Export and promote a local bundle"
just rebuild-artifacts models
just serve-with-assets
```

This explicit rebuild refreshes project and packaged player-state files. For a scratch export use `just train-local`, then `just promote-local` to write a local pointer.

Promotion writes `models/promoted_model.json`. Schema v2 binds the three
filenames to the selected model kind, recorded training version, model SHA-256,
metadata SHA-256, model-card SHA-256, and selection reason. Promotion recomputes
selection from candidate metrics and validates every identity without
unpickling the model.

`serve-with-assets` sets `ACEBET_MODEL_DIR` to `models/` and
`ACEBET_DATA_DIR` to `data/`, so this promoted pointer and the refreshed player
state become the production serving inputs instead of merely being local build
outputs. Startup compares the mounted player state to the digest in model
metadata. When using a custom dataset, generate the state from that same file
with `acebet.dataprep.dataprep --input-path … --player-stats-output-path …`
in the same locked environment as training.

Schema-v1 metadata and pointers are deliberately unsupported. Regenerate and
promote them with the current workflow.

## Loading and trust boundary

Directory loading requires `promoted_model.json`; there is no newest-filename
fallback. A direct joblib path remains available to trusted tooling, but its
complete v2 sidecars are still mandatory.

Before `joblib.load`, the loader validates the pointer hashes, metadata schema,
feature contract, recomputed selection, selected kind, model digest, and
agreement between the pointer and metadata about the training version.
After deserialization it verifies the Python type and fitted feature
contract for Elo or LightGBM.

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


## Elo timing probe


The prepared file contains Elo ratings and odds supplied by the upstream data.
The original rating code and point-in-time records are unavailable. Across
75,072 pairs of consecutive appearances by the same player, excluding dates
with multiple appearances by that player, Elo rose after 99.99% of wins and
fell after 99.91% of losses. The change aligned with the next match’s outcome
in 47.91% of wins and 50.21% of losses. This pattern is consistent with the
current row storing a rating from before its own match.

Across 32,663 matches with valid Pinnacle prices, Elo log loss was 0.6029
and the price-implied baseline was 0.5655 after normalizing inverse odds to
sum to one. Their probability correlation was 0.8716. These checks support
the pre-match interpretation. They leave the upstream generation and timing
of ranks and other inputs unverified.

```bash
uv run python -c 'from acebet.dataprep.dataprep import prepare_data, summarize_elo_causal_evidence; print(summarize_elo_causal_evidence(prepare_data()))'
```

## Serving state

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


Set `ACEBET_MAX_SNAPSHOT_AGE_DAYS` to reject snapshots older than your limit. With it unset, the service logs a warning for snapshots older than 60 days. Age cannot establish forecast quality. [Exact request and response](../api/sync-endpoints.md)
