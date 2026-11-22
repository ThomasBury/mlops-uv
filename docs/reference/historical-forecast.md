# Historical replay reference

These interfaces operate on the 2021 replay. The offline entry point and the localhost HTTP handler use the same `ForecastRequest` and `FrozenForecast.predict`. The [packaged API](../api/sync-endpoints.md) accepts a different request.

## Requests and responses

| Field | Contract |
| --- | --- |
| `fixture_id` | `F01` or a recorded final-period match ID. |
| `p1_id`, `p2_id` | The two players from that fixture, in either order. All three IDs are strings of 1–64 ASCII letters, digits or hyphens. |
| `surface` | Exactly `Clay`, `Grass` or `Hard`, matching the record. |
| `cutoff` | ISO timestamp with a `T` and UTC offset, normalized to UTC. F01 uses noon on June 10, 2030. Historical fixtures use the last second of the prior UTC day. |
| Extra fields | Rejected. The service derives the ratings. |

```json
{"fixture_id":"F01","p1_id":"P01","p2_id":"P02","surface":"Clay","cutoff":"2030-06-10T12:00:00Z"}
```

The reply echoes the normalized request. `p1_win_probability` is for the first requested ID. `elo_gap` is that player’s rating minus the opponent’s. `model_sha256` identifies the fitted pipeline bytes. `history_sha256` covers the canonical results. `feature_state_sha256` covers the F01 trace or, for an archived fixture, the history replayed at the stated cutoff.

| Method | Path | Result |
| --- | --- | --- |
| POST | `/forecast` | 200 with the reply, 422 for an invalid request or fixture mismatch, 503 if assets failed startup checks. |
| GET | `/healthz` | 200 while the worker can answer. |
| GET | `/readyz` | 200 when a verified forecaster is loaded, otherwise 503. |
| GET | `/metrics` | Prometheus text when enabled, otherwise 404. |

The historical worker has no token flow. The course binds it to loopback. The authenticated packaged app is a separate process.

## Commands and files

From the checkout, use the locked Python 3.12 environment with the `mlops` group and `tracing` extra. In this table each Python module is invoked with `.venv/bin/python -m`. Paths in capitals are arguments.

| Module and arguments | Output or decision |
| --- | --- |
| `acebet.dataprep.historical_results CSV RESULTS` | Writes validated Parquet. Standard output gives source hash, logical content hash and row count. |
| `acebet.dataprep.historical_replay [RESULTS]` | Without a path, prints F01’s feature trace. With Parquet, compares as-of and leaky Elo. |
| `acebet.train.historical_evaluation compare RESULTS` | Development scores by fold and a random-split timing diagnostic. |
| `acebet.train.historical_evaluation train RESULTS BUNDLE` | Creates a new directory. `selection.json` links input and source hashes, dependency versions, feature policy, seed, settings and model hash. `membership.csv` records split roles and availability. `search.csv` records the five trials. `model.joblib` is the fitted pipeline. |
| `acebet.train.historical_evaluation evaluate RESULTS BUNDLE` | Writes `final/forecasts.csv` with one row per match and `final/evaluation.json` with counts, metrics, intervals and run IDs. |
| `acebet.historical_forecast RESULTS BUNDLE` | Reads one request as JSON from stdin, writes one reply to stdout. |
| `acebet.historical_monitoring quality LOG RESULTS AS_OF` | Scores recorded forecasts against results available by the reporting clock. |
| `acebet.historical_monitoring gate FINAL --candidate NAME --incumbent NAME` | Recomputes the paired interval. Use `prevalence`, `elo` or `logistic_tuned`. Exit 1 means rejection. |
| `acebet.historical_monitoring review RESULTS BUNDLE` | Emits the verified F01 reply, selection and evaluation identities, and the candidate-versus-Elo decision. |
| `acebet.historical_monitoring promote RESULTS BUNDLE` | Writes `promotion.json` only on acceptance. A supervisor separately selects the bundle to serve. |

`acebet.app.historical:create_app` is the Uvicorn factory. Set `ACEBET_FORECAST_RESULTS` to the Parquet file and `ACEBET_FORECAST_BUNDLE` to the model directory. [Local serving exercise](../tutorial/serve-the-frozen-forecast.md) · [Container target](../tutorial/reproduce-the-forecast-runtime.md)

## Troubleshooting

| Observation | Inspect first |
| --- | --- |
| Raw SHA-256 mismatch | Compare the bytes to the pinned extract below, including BOM and line endings. Edit a separate sample to test normalization. |
| Output directory already exists | Training and evaluation protect earlier evidence. Inspect the existing run and choose a fresh destination for a new one. |
| 422 on an archived request | Compare ID pair, surface and offset-aware cutoff with its row in `forecasts.csv`. |
| Live but unready, or model checksum differs | Read `forecast_assets_invalid` in the server log. Check that selection, model and history belong to the same run and inference dependencies match. Restore a trusted pair. |
| One empty calibration period | Check cutoff month, retained log and label arrival time. A missing label has no loss. |
| Successful candidate job, no approval | Inspect `validation.json` and the upper endpoint of its loss interval. |

## Source snapshot

| Provenance field | Recorded value |
| --- | --- |
| Upstream file | [Valuebetennis 2021 CSV](https://www.valuebetennis.com/datasets/valuebetennis-matchs-2021.csv) |
| Observed export revision | Retrieved 2026-11-20 UTC. HTTP `Last-Modified` was `Sun, 08 Nov 2026 05:45:03 GMT` and `ETag` was `"908c48-65fbd58c59141"`. |
| Full export SHA-256 | `ea30ae06fab7734ada518b974aacf55bea3cde41ff569f1135c9d05f79cc9aee` |
| Extraction rule | Keep the UTF-8 BOM and header, then select rows with `genre=atp`, `categorie` equal to `Main tour` or `Masters`, and integer `tour` from 4 to 12 inclusive. The pinned file contains 2,091 rows from the 63,728-row export. |
| Extract SHA-256 | `ca16daa1c89b085680152dc41924cad7e2791775c23d68a083c2c0237cdf0598` |

AceBet distributes this filtered extract of *Valuebetennis, Tennis results and odds since 2021* under CC BY 4.0. The derived Parquet embeds provenance. The full export supplies settled results. Its publication delay and the availability of the recorded surface at forecast time are assumptions of this replay. [Provider definitions](https://www.valuebetennis.com/en/donnees.htm) · [License](https://creativecommons.org/licenses/by/4.0/)

## Repeat the run

Use the scratch directory and `first` bundle from lessons 2–6. Set `snapshot=data/raw/valuebetennis-2021-atp.csv` in the checkout root. Use fresh output paths on each repetition.

Declare the replay tolerances before retraining. The match IDs, labels, selected
parameters, and calibration counts must agree exactly. Probabilities, scores,
and interval endpoints must agree to an absolute tolerance of `1e-8` with
zero relative tolerance. Use the same locked environment and check the
recorded source and dependency identities. Artifact hashes identify a
particular run's files. The numerical comparison is the repeatability check.

```bash
.venv/bin/python -m acebet.dataprep.historical_results \
  "$snapshot" "$work/rebuilt.parquet"
.venv/bin/python -m acebet.train.historical_evaluation train \
  "$work/rebuilt.parquet" "$work/repeat"
.venv/bin/python -m acebet.train.historical_evaluation evaluate \
  "$work/rebuilt.parquet" "$work/repeat"
.venv/bin/python - "$work/first" "$work/repeat" <<'PY'
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

first, repeat = map(Path, sys.argv[1:])
def pair(filename):
    return [json.loads((path / filename).read_text()) for path in (first, repeat)]

def close(left, right):
    np.testing.assert_allclose(left, right, atol=1e-8, rtol=0)

left, right = pair("selection.json")
for key in ("source_sha256", "data_sha256", "development_sha256", "sources_sha256",
            "dependencies", "best_params", "config"):
    assert left[key] == right[key], key
for name in left["mean_log_loss"]:
    close(left["mean_log_loss"][name], right["mean_log_loss"][name])
for key in ("p1_win_probability", "logistic_p1_win_probability", "unseen_surface_p1_win_probability"):
    close(left["fixture"][key], right["fixture"][key])
pd.testing.assert_frame_equal(pd.read_csv(first / "membership.csv"), pd.read_csv(repeat / "membership.csv"))
pd.testing.assert_frame_equal(pd.read_csv(first / "search.csv"), pd.read_csv(repeat / "search.csv"), atol=1e-8, rtol=0)

a, b = [pd.read_csv(path / "final/forecasts.csv") for path in (first, repeat)]
pd.testing.assert_frame_equal(a[["match_id", "label"]], b[["match_id", "label"]])
models = ["prevalence", "elo", "logistic_tuned"]
close(a[models], b[models])
left, right = pair("final/evaluation.json")
assert left["matches"] == right["matches"] == len(a) == 369
for name in models:
    close(left["log_loss"][name], right["log_loss"][name])
    for observed, reproduced in zip(left["calibration"][name], right["calibration"][name], strict=True):
        for field in ("lower", "upper", "matches"):
            assert observed[field] == reproduced[field]
        if observed["matches"]:
            for key in ("mean_probability", "observed_win_rate"):
                close(observed[key], reproduced[key])
key = "log_loss_delta_candidate_minus_baseline"
for name in ("prevalence", "elo"):
    assert left[key][name]["weeks"] == right[key][name]["weeks"] == 9
    for metric in ("mean_delta", "interval_95"):
        close(left[key][name][metric], right[key][name][metric])
print("Reproduced F01 and 369 final forecasts within tolerance.")
PY
```
