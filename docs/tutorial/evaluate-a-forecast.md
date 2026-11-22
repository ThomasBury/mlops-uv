# 6. Freeze a candidate and judge the evidence

The first comparisons favor Elo. Can a different amount of regularization help logistic regression? Use the earlier folds to answer that question. Declare the final decision rule now so you cannot change it after seeing September’s results.

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
.venv/bin/python -m acebet.train.historical_evaluation train \
  "$work/matches.parquet" "$work/first"
```

The search chooses `C = 0.3` with mean fold log loss `0.68085`. Elo still
has the lower development score. Keep the logistic candidate frozen for the
final comparison. `search.csv` records the five trials using scikit-learn's
negative log loss scorer, so higher values rank first in that file.

The search uses `RandomizedSearchCV` with `cv=folds` and `refit=False`. Every trial fits the entire pipeline on each training window. After selecting the penalty, `train_candidate` fits a fresh pipeline once on all mature development rows. The saved `model.joblib` therefore includes a scaler and surface encoder fitted to that population. For F01 its estimate is 0.46910. Changing only the surface to an unseen category produces an all-zero surface encoding and an estimate of 0.47672.

Open `first/search.csv` to see all five trials. Select one ID in `first/membership.csv` and follow its role through a fold and `final_refit`. `first/selection.json` links that record to the source and history hashes, code, dependency versions, chosen parameter and model bytes. Trust the producer before loading a Joblib file.

## Open the final period

```bash
.venv/bin/python -m acebet.train.historical_evaluation evaluate \
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

## Check the decision from individual forecasts

The final run carries `selection_run_id` as a tag. Follow it back to the development search in MLflow. Both runs live in `$work/mlflow.db` and their artifacts under `$work/mlartifacts`. The selection run owns the fitted model and fold membership. The final run owns the row-level predictions and scores. Use the recorded winner in each row to check the difference independently.

Save this as `score-pairs.py` in your scratch directory.

```python
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from mlflow import MlflowClient

work = Path(sys.argv[1])
bundle = work / "first"
selection = json.loads((bundle / "selection.json").read_text())
final = json.loads((bundle / "final/evaluation.json").read_text())
client = MlflowClient(tracking_uri=f"sqlite:///{work}/mlflow.db")
assert client.get_run(final["run_id"]).data.tags["selection_run_id"] == selection["run_id"]
rows = pd.read_csv(bundle / "final/forecasts.csv")
assert len(rows) == rows.match_id.nunique() == 369

def loss(probability):
    return -np.log(np.where(rows.label == 1, probability, 1 - probability))

print((loss(rows.logistic_tuned) - loss(rows.elo)).mean())
bins = pd.cut(rows.logistic_tuned, [0, .2, .4, .6, .8, np.inf], right=False)
print(rows.groupby(bins, observed=False).agg(
    matches=("label", "size"), forecast=("logistic_tuned", "mean"), wins=("label", "mean"),
))
```

```bash
.venv/bin/python "$work/score-pairs.py" "$work"
```

Do the counts support a claim about predictions above 80%? What would you do if someone wanted to tune `C` again after seeing this table? Reserve another later period for that new decision. These 369 matches are settled results, scored under the assumed release delay. Live schedule coverage and actual input arrival times remain unknown.

Keep `first` and `matches.parquet` together for the [serving exercise](./serve-the-frozen-forecast.md). To repeat training independently, use the [full numerical comparison](../reference/historical-forecast.md#repeat-the-run).
