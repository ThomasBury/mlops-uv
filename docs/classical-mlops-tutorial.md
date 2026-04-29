# Classical MLOps Tutorial

This repository demonstrates a small but executable MLOps loop for a tabular
classification problem: predict the winner of an ATP match from prepared match
history and player-state features.

## 1. Reproducible Environment

```bash
uv sync
```

The lockfile is the reproducibility boundary for local work, CI, Docker, and
the Zensical docs build.

## 2. Prepared Data Contract

The canonical prepared dataset is `data/atp_data_production.feather`.

```bash
just data-check
```

`acebet.dataprep.prepare_data` requires at least:

- `date`
- `p1`
- `p2`
- `rank_p1`
- `rank_p2`
- `elo_p1`
- `elo_p2`
- `proba_elo`
- `target`

The dataprep step sorts by date and recomputes deterministic features such as
`year`, `month`, `day`, `rank_diff`, and `best_ranked`.

## 3. Player-State Artifacts

```bash
just feature-state
```

That command writes three player-state tables:

- `data/latest_player_stats.feather` for project-root production use
- `src/acebet/data/latest_player_stats.feather` as a packaged production fallback
- `src/acebet/data/latest_player_stats_sample.feather` as the packaged sample used first in testing mode

These files are built from the latest appearance of each player in the prepared
dataset.

## 4. Shared Feature Contract

Training and serving meet at three functions in `src/acebet/features.py` and
`src/acebet/app/dependencies/predict_winner.py`:

- `select_model_features` drops forbidden columns such as labels, final scores, odds, and comments.
- `add_derived_match_features` and `build_match_feature_row` assemble deterministic features.
- `align_features_to_model` reorders and pads inference-time columns to match the trained model.

One detail matters for correctness: `proba_elo` is required in the prepared
training dataset, but online scoring recomputes it from `elo_p1` and `elo_p2`
instead of reading it from a stored match row.

## 5. Train And Export

```bash
just train
```

The training script loads the prepared dataset, filters the configured date
window, uses a `TimeSeriesSplit`, fits a LightGBM model inside a scikit-learn
pipeline, evaluates the latest fold, and exports `models/model_*.joblib`.

No `model_*.metadata.json` sidecar is produced.

Optional MLflow logging is available with:

```bash
just train-mlflow
```

That workflow is separate from the default environment. The recipe first runs
`uv sync --group mlops`, then starts training with `--enable-mlflow`.

By default the training script uses:

- tracking URI: `sqlite:///mlflow.db`
- experiment name: `acebet`
- local artifact root: `mlartifacts/`

When enabled, the code logs training parameters, evaluation metrics, the
scikit-learn model, and the exported joblib artifact if one was written. The
training CLI also supports custom `--tracking-uri`, `--experiment-name`, and
`--artifact-root` values for local or remote tracking setups.

## 6. Serve Predictions

```bash
just serve
```

At startup the API preloads both runtime modes:

- Production mode prefers `data/latest_player_stats.feather`, then falls back to packaged production data.
- Testing mode prefers `src/acebet/data/latest_player_stats_sample.feather`, then packaged production data, then project data.
- Model lookup is the same in both modes: `models/`, then `src/acebet/data/`, then the project root.

`POST /predict/` builds an online feature row from player names, match date, and
optional match context. The request field is `best_of`, but the internal feature
row translates that into the `"best of"` column name expected by the training
data. That translation is an implementation detail, not a separate public API.

## 7. Test And Review

```bash
just test
just manual
just review
```

The automated tests cover auth, prediction, prepared-data checks, feature
assembly, and model export. `just manual` exercises the live HTTP flow. `just
review` chains the full local validation path.

## 8. Package And Containerize

```bash
uv build
docker build -t acebet .
docker run --rm -p 8000:80 acebet
```

The Docker image copies the repository, runs `uv sync --frozen --no-cache`, and
starts `fastapi run src/acebet/app/main.py --host 0.0.0.0 --port 80`.
