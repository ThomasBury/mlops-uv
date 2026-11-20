---
title: Decide whether a forecast improves on Elo
description: Compare models in chronological order, freeze a candidate, and test it on later matches.
---

# Decide whether a forecast improves on Elo

You have [match results and pre-match Elo ratings](./rebuild-match-history.md).
Should you replace Elo with a model that learns from the Elo gap and court
surface? Use earlier matches to choose the candidate and later matches to
check whether the improvement holds up. Keep [F01](./forecast-contract.md) as
a known input for replaying the fitted pipeline.

## Compare on earlier matches

From the checkout root, rebuild the M3 extract in a scratch directory. These
commands assume the Python 3.12 environment from the previous lessons.

```bash
work="$(mktemp -d)"
snapshot=data/raw/valuebetennis-2021-atp.csv
uv run --locked python -m acebet.dataprep.historical_results "$snapshot" "$work/matches.parquet"
uv run --locked python -m acebet.train.historical_evaluation compare "$work/matches.parquet"
```

The three validation periods are March–April, May–June, and July–August
2021. In each fold, the training window starts at the beginning of the
history and ends before validation starts. A match can train the model only
if its label was already available at the fold boundary. A validation label
can inform candidate selection only if it is available by the final cut on
August 31 at `23:59:59 UTC`. These rules use M3's assumed result availability.
Match IDs stay together, including all matches on the same event day within a
validation window. Players can appear repeatedly because the question is how
to forecast the next match, including matches between returning players.

**First comparison.** Training prevalence predicts the fraction of training
matches won by the player with the smaller ID. Elo converts the two ratings
available at each fixture's cutoff into a probability. Logistic regression
learns a probability from the Elo difference and surface. Each training fold
fits its own scaler for the Elo gap and its own one-hot encoder for surface.
The `Pipeline` fits both transformations before fitting the classifier, and
reuses those transformations during validation. [scikit-learn on data leakage](https://scikit-learn.org/stable/common_pitfalls.html)

**Second comparison.** Leaky Elo updates both ratings with the current winner
and then scores that same match. Compare it with the repaired as-of Elo on
the same validation fixtures. Mean log loss gives each fold equal weight.
Lower values are better.

| Development forecast | Mean fold log loss |
| --- | ---: |
| Training prevalence | 0.69381 |
| As-of Elo | 0.67036 |
| Logistic regression, default `C` | 0.68092 |
| Elo after seeing the current winner | 0.58619 |

Find `random_split_diagnostic` in the output. Its model used a training label
that became available on August 31, after all 444 of its validation cutoffs.
Its loss is about `0.66743`, calculated on a different set of matches. The
timestamp comparison is the evidence that this split permits learning from
the future.

## Freeze the candidate and its record

Before opening the final scores, choose the logistic regression setting
with the lowest mean log loss across the three folds. Draw five `C` values
from the nine recorded options with seed `42`. `C` is the inverse strength
of the coefficient penalty. On an exact tie, take the first sampled setting.
Refit that setting once on all 1,487 mature development matches. Carry
training prevalence and as-of Elo forward as prespecified baselines. Judge
the candidate better than Elo only if the entire paired interval for
candidate minus Elo loss is below zero.

```bash
uv run --locked --group mlops python -m acebet.train.historical_evaluation train \
  "$work/matches.parquet" "$work/first"
```

The search chooses `C = 0.3` with mean fold log loss `0.68085`. Elo still
has the lower development score. Keep the logistic candidate frozen for the
final comparison. `search.csv` records the five trials using scikit-learn's
negative log loss scorer, so higher values rank first in that file.

`selection.json` records the source and canonical data hashes, code file
hashes, installed package versions, settings, seed, development scores, and
artifact hashes. `membership.csv` lists training and validation match IDs,
event dates, forecast cutoffs, and label availability for each temporal
fold and the deliberately flawed `random_diagnostic` split. The
`final_refit` rows show what trained the frozen `model.joblib`, which includes
the fitted scaler and encoder. Open `membership.csv` and trace a returning
player across a fold boundary. Check the latest training-label timestamp
against the earliest validation cutoff. Only load Joblib bundles you trust.

The `fixture` record carries the noon cutoff for F01, its contributing result,
and the frozen model's `0.46910` win probability for player `P01` on clay.
The second prediction changes only the surface to `Unseen`. Its surface
encoding is all zeros and its probability is `0.47672`. The numeric Elo input
still contributes. This demonstrates how the pipeline handles an unfamiliar
category. F01 is synthetic and stays out of the historical scores.

MLflow records this run in the local SQLite database `$work/mlflow.db` with
copies of the artifacts under `$work/mlartifacts`. The UI is optional. The
command below opens it at `http://127.0.0.1:5000`.
[MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)

```bash
uv run --locked --group mlops mlflow server --host 127.0.0.1 --backend-store-uri "sqlite:///$work/mlflow.db"
```

## Open the final period

```bash
uv run --locked --group mlops python -m acebet.train.historical_evaluation evaluate \
  "$work/matches.parquet" "$work/first"
```

This command verifies the frozen inputs and scores the same 369 fixtures for
all three forecasts. The model and preprocessing stay fixed. The ratings
can incorporate earlier final-period results once those results are available
at a later fixture's cutoff. Inspect `first/final/forecasts.csv` for the
held-out IDs, model inputs, probabilities, and eventual labels.
`first/final/evaluation.json` and the linked MLflow run identify the frozen
selection and report these results.

| Final forecast | Log loss | Candidate minus baseline, mean [95% interval] |
| --- | ---: | --- |
| Training prevalence | 0.69299 | −0.06820 [−0.08774, −0.03916] |
| As-of Elo | 0.62609 | −0.00130 [−0.00250, 0.00057] |
| Tuned logistic regression | 0.62479 | |

For each match, subtract the baseline's loss from the candidate's loss.
Resample the nine nonempty Monday–Sunday UTC weeks with replacement 2,000
times and calculate the mean per match in each draw. This keeps the two
forecasts paired and keeps matches sharing a week together. It partly
accounts for tournament scheduling and repeated players. A tournament can
span weeks, and nine weeks from one season give a rough interval. The final
comparison favors the candidate over prevalence. The small edge over Elo
is inconclusive because the interval includes zero. Keep Elo as the benchmark.

Calibration compares the average forecast with the observed win rate in
five fixed probability bins. For the logistic candidate, the `0.6` to
`0.8` bin has 85 matches, an average forecast of `0.677`, and a win rate of
`0.741`. The highest bin has only seven matches. Treat that gap as a lead
for investigation. Inspect counts across all bins before diagnosing a
systematic calibration problem.

## Rebuild and reproduce the evidence

Declare the replay tolerances before retraining. The match IDs, labels, selected
parameters, and calibration counts must agree exactly. Probabilities, scores,
and interval endpoints must agree to an absolute tolerance of `1e-8` with
zero relative tolerance. Use the same locked environment and check the
recorded source and dependency identities. Artifact hashes identify a
particular run's files. The numerical comparison is the repeatability check.

```bash
uv run --locked --group mlops python -m acebet.dataprep.historical_results \
  "$snapshot" "$work/rebuilt.parquet"
uv run --locked --group mlops python -m acebet.train.historical_evaluation train \
  "$work/rebuilt.parquet" "$work/repeat"
uv run --locked --group mlops python -m acebet.train.historical_evaluation evaluate \
  "$work/rebuilt.parquet" "$work/repeat"
uv run --locked --group mlops python - "$work/first" "$work/repeat" <<'PY'
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
for key in ("data_sha256", "sources_sha256", "dependencies", "best_params", "config"):
    assert left[key] == right[key], key
for name in left["mean_log_loss"]:
    close(left["mean_log_loss"][name], right["mean_log_loss"][name])
for key in ("p1_win_probability", "logistic_p1_win_probability", "unseen_surface_p1_win_probability"):
    close(left["fixture"][key], right["fixture"][key])
pd.testing.assert_frame_equal(pd.read_csv(first / "membership.csv"), pd.read_csv(repeat / "membership.csv"))

a, b = [pd.read_csv(path / "final/forecasts.csv") for path in (first, repeat)]
pd.testing.assert_frame_equal(a[["match_id", "label"]], b[["match_id", "label"]])
models = ["prevalence", "elo", "logistic_tuned"]
close(a[models], b[models])
left, right = pair("final/evaluation.json")
for name in models:
    close(left["log_loss"][name], right["log_loss"][name])
    for observed, reproduced in zip(left["calibration"][name], right["calibration"][name]):
        assert observed["matches"] == reproduced["matches"]
        if observed["matches"]:
            for key in ("mean_probability", "observed_win_rate"):
                close(observed[key], reproduced[key])
key = "log_loss_delta_candidate_minus_baseline"
for name in ("prevalence", "elo"):
    for metric in ("mean_delta", "interval_95"):
        close(left[key][name][metric], right[key][name][metric])
print("Reproduced F01 and 369 final forecasts within tolerance.")
PY
```

For serving in M5, keep the frozen bundle, canonical history, and F01 trace
together. Use the model and data hashes to identify what produced a forecast.
The [serving exercise](./serve-the-frozen-forecast.md) uses this candidate
while the comparison with Elo remains inconclusive.
