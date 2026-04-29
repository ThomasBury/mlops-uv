# Training and Inference

This page covers the artifact flow and the runtime feature contract used by the
current codebase.

## Training Flow

```bash
just data-check
just feature-state
just train
```

That sequence does three distinct jobs:

- `just data-check` validates `data/atp_data_production.feather`.
- `just feature-state` derives player-state tables for project and packaged use.
- `just train` exports a timestamped model to `models/model_*.joblib`.

The training path does not create a `model_*.metadata.json` sidecar.

Optional MLflow tracking is available through:

```bash
just train-mlflow
```

That recipe opts into the `mlops` dependency group before training. In the
default local setup, MLflow uses:

- `sqlite:///mlflow.db` as the tracking URI
- `acebet` as the experiment name
- `mlartifacts/` as the local artifact root

The training code logs run parameters, metrics, the sklearn model, and the
exported joblib artifact when one exists. If you need a different backend, the
underlying CLI supports `--tracking-uri`, `--experiment-name`, and
`--artifact-root`.

## Feature Contract

The shared contract is implemented across:

- `select_model_features`
- `build_match_feature_row` and `add_derived_match_features`
- `align_features_to_model`

Operationally, that means:

- Forbidden columns such as `target`, final scores, bookmaker odds, and `comment` are dropped before fitting or inference.
- The prepared training dataset must already contain `proba_elo`.
- Online inference recomputes `proba_elo` from the latest player ELO values.
- Request field `best_of` is translated internally to the `"best of"` column expected by the model features.

## Inference Flow

Start the API:

```bash
just serve
```

Get a token:

```bash
just token-json
```

Exercise prediction in testing mode:

```bash
just predict-testing
```

Exercise prediction in production mode:

```bash
just predict-production
```

Both prediction modes authenticate the same way. The `testing` flag only changes
which runtime assets are preferred.

## Asset Resolution

- Production player stats: `data/latest_player_stats.feather`, then packaged production stats.
- Testing player stats: packaged sample stats, then packaged production stats, then project data.
- Models: `models/`, then `src/acebet/data/`, then the project root.

If you train a fresh model locally, that exported artifact is preferred over the
packaged fallback on the next API startup.

## Manual HTTP Example

```bash
curl -X POST http://127.0.0.1:8000/token \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=johndoe&password=secret'
```

```json
{
  "p1_name": "Fognini F.",
  "p2_name": "Jarry N.",
  "date": "2018-03-04",
  "surface": "Clay",
  "round": "Final",
  "testing": false
}
```
