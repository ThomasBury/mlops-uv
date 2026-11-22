# M4c. Judge the frozen candidate

Optuna picked the setting that scored best on the three development folds. You recovered the fitted pipeline in [M4b](m4b.md). Does it improve on Elo for later matches? Keep the decision rule from M4a and open the held-out period once.

## Score the recovered model with fitting disabled

Continue from the same shell with `results`, `bundle`, `m4`, and `python4` set. The bundle path should point to `$m4/recovered`. Use the CLI entry point inside a guard that raises if the evaluation calls fit.

If you are following the versioned checkpoints, run `git switch --detach course-v1-m4c` in your clean course checkout. Use the candidate produced by `course-v1-m4b`.

```bash
"${python4[@]}" - "$results" "$bundle" <<'PY'
import sys
from unittest.mock import patch

from sklearn.pipeline import Pipeline

from acebet.train.historical_evaluation import main

with patch.object(Pipeline, "fit", side_effect=AssertionError("Final evaluation attempted to fit")):
    main(["evaluate", *sys.argv[1:]])
PY
```

`evaluate_candidate` verifies the canonical history, training code and dependency versions, and the hashes of the frozen bundle. It loads the fitted pipeline and selects the 369 matches from September 1 onward. Training prevalence and the logistic pipeline stay fixed. As-of Elo updates when earlier results become available for a later match. All three forecasts use the same fixtures and eventual outcomes. [Evaluation code](../src/acebet/train/historical_evaluation.py)

```python
predictions = {
    "prevalence": np.full(len(final), selection["refit_prevalence"]),
    "elo": final["elo"].to_numpy(),
    "logistic_tuned": model.predict_proba(final[FEATURES])[:, 1],
}
```

The forecasts go to `final/forecasts.csv`. The scorecard in `final/evaluation.json` identifies the selection run, selection record, model bytes, and forecast CSV. The evaluation is also an MLflow run tagged with the selection run ID.

| Forecast | Final log loss | Candidate minus baseline, mean [95% interval] |
| --- | ---: | --- |
| Training prevalence | 0.69299 | −0.06351 [−0.08013, −0.03904] |
| As-of Elo | 0.62609 | +0.00340 [+0.00019, +0.00547] |
| Tuned logistic regression | 0.62949 | |

A colleague claims the development winner is ready to replace Elo because it beat the untuned logistic model. Another sees the held-out mean difference of `+0.00340` and says that alone settles the question. Use the interval and the agreed rule to answer both. Positive differences favor Elo, and the entire interval is positive in this replay. Keep Elo. The candidate improves on training prevalence. Changing `C` after looking at these scores would make this period part of development. Reserve a later period to judge a change.

## Recompute the paired result and inspect calibration

For each fixture, subtract the baseline's loss from the candidate's loss. The two forecasts in a pair face the same outcome. Matches in one tournament week can share players and conditions, so the interval resamples whole nonempty Monday–Sunday UTC weeks. Each of the 2,000 draws samples nine weeks with replacement. Divide the total sampled loss difference by the total sampled matches to weight each match equally. Use the 2.5th and 97.5th percentiles. Nine weeks from one season give a rough interval, and players can connect matches across weeks.

Calibration describes how forecast probabilities line up with observed win frequencies. Read the number of matches in each bin along with those frequencies. Reconstruct both the paired interval and the calibration summary from the exported forecasts.

```bash
"${python4[@]}" - "$bundle" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

bundle = Path(sys.argv[1])
selection_path = bundle / "selection.json"
selection = json.loads(selection_path.read_text())
evaluation = json.loads((bundle / "final/evaluation.json").read_text())
forecasts = bundle / "final/forecasts.csv"
assert sha256(selection_path.read_bytes()).hexdigest() == evaluation["selection_sha256"]
assert sha256(forecasts.read_bytes()).hexdigest() == evaluation["forecasts_sha256"]
assert evaluation["model_sha256"] == selection["artifacts_sha256"]["model.joblib"]
assert evaluation["selection_run_id"] == selection["run_id"]

rows = pd.read_csv(forecasts, dtype={"match_id": str})
config = selection["config"]
assert len(rows) == rows["match_id"].nunique() == evaluation["matches"] == 369
assert (rows["event_date"] >= config["final_start"]).all()

def loss(column):
    eps = np.finfo(float).eps
    p = rows[column].to_numpy().clip(eps, 1 - eps)
    return -np.log(np.where(rows["label"] == 1, p, 1 - p))

def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)

for name, recorded in evaluation["log_loss"].items():
    close(loss(name).mean(), recorded)

candidate_loss = loss("logistic_tuned")
weeks = pd.to_datetime(rows["event_date"]).dt.to_period("W-SUN")
for baseline, report in evaluation["log_loss_delta_candidate_minus_baseline"].items():
    delta = candidate_loss - loss(baseline)
    blocks = pd.Series(delta).groupby(weeks).agg(["sum", "size"])
    assert len(blocks) == report["weeks"] == 9
    picks = np.random.default_rng(config["seed"]).integers(
        len(blocks), size=(config["bootstrap_draws"], len(blocks))
    )
    totals = blocks.to_numpy()[picks].sum(axis=1)
    interval = np.quantile(totals[:, 0] / totals[:, 1], [.025, .975])
    close(delta.mean(), report["mean_delta"])
    close(interval, report["interval_95"])
    print(baseline, delta.mean(), interval)

edges = [*config["calibration_bin_edges"][:-1], np.inf]
bins = pd.cut(rows["logistic_tuned"], edges, right=False)
calibration = rows.groupby(bins, observed=False).agg(
    matches=("label", "size"), forecast=("logistic_tuned", "mean"), wins=("label", "mean"),
)
for row, recorded in zip(calibration.itertuples(), evaluation["calibration"]["logistic_tuned"], strict=True):
    assert row.matches == recorded["matches"]
    if row.matches:
        close(row.forecast, recorded["mean_probability"])
        close(row.wins, recorded["observed_win_rate"])
print(calibration)
PY
```

In the 0.6–0.8 bin, 81 forecasts average about `0.668` and the first player won about `0.741` of the matches. The bin at 0.8 and above has two matches. One different outcome would move its observed win rate by 50 percentage points. The archive covers completed matches and uses assumed availability times. The score describes the eligible fixtures in this replay.

## Hand off a forecast you can verify

For independent reruns, require the same data and source identities, configuration, selected parameters, fixture IDs, labels, and calibration counts. Allow absolute error of `1e-8` with zero relative tolerance for probabilities, losses, and interval endpoints. Run IDs identify experiments and model hashes identify the bytes used by a particular run.

Keep the Parquet with the recovered bundle and its final report. Save an offline F01 prediction for the service comparison.

```bash
"${python4[@]}" -m acebet.historical_forecast \
    "$results" "$bundle" > "$m4/f01.json" <<'JSON'
{"fixture_id":"F01","p1_id":"P01","p2_id":"P02","surface":"Clay","cutoff":"2030-06-10T12:00:00Z"}
JSON
cat "$m4/f01.json"
```

The reply should give a first-player probability near `0.475945` and identify the model, history, and feature state. Take that reply and the batch forecast for match `424816` to [M5](m5.md) and check that the served forecast agrees.
