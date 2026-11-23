# M4a. Tune regularization with Optuna

Elo had the lowest development loss in [M3](m3.md). Logistic regression also sees court surface, but it has to estimate those coefficients from a short history. Can stronger regularization improve the logistic model on the same three development folds?

## Decide what a trial can see

Use positive log loss as the objective and minimize the mean of the three fold losses. Give each validation window equal weight, even when it contains a different number of matches. Reserve fixtures dated September 1 onward for [M4c](m4c.md). Before seeing their scores, agree to keep Elo unless the entire 95% paired interval for candidate minus Elo loss falls below zero.

Use Bash, uv, and Python 3.12 from the repository root. If you have M3's canonical Parquet and comparison directory, keep `results` and set `comparison="$work/comparison"`. For direct entry, rebuild them from the pinned snapshot.

If you are following the versioned checkpoints, switch your clean course checkout to `course-v1-m4a` with `git switch --detach course-v1-m4a`. Keep the M3 outputs outside that checkout.

```bash
m4="$(mktemp -d)"
python4=(uv run --locked --python 3.12 --group mlops python)
```

```bash
# Rebuild these inputs only if you need a fresh copy of the M3 handoff.
results="$m4/matches.parquet"
comparison="$m4/comparison"
"${python4[@]}" -m acebet.dataprep.historical_results \
    data/raw/valuebetennis-2021-atp.csv "$results"
"${python4[@]}" -m acebet.train.historical_evaluation \
    compare "$results" --output "$comparison"
```

## Follow one trial through the folds

One trial proposes a setting for `C`, the inverse of regularization strength. Smaller values shrink the coefficients more. At very small `C`, the Elo gap and surface have less influence on the forecast. Search from `0.001` to `10` on a logarithmic scale so each order of magnitude has the same width. Optuna's `TPESampler` starts with five random trials, then uses their scores to guide later proposals. Run 12 trials sequentially with seed 42. [Optuna study and trial guide](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html)

In the objective, one proposed `C` goes into three fresh pipelines. The scaler, surface encoder, and classifier learn from each fold's own earlier rows. Validation calls `predict_proba` on the pipeline fitted for that fold. [Search implementation](../src/acebet/train/historical_evaluation.py)

```python
c = trial.suggest_float("C", *C_BOUNDS, log=True)
# Inside the loop over the M3 folds:
model = logistic_pipeline(c).fit(train[FEATURES], train["label"])
probability = model.predict_proba(valid[FEATURES])[:, 1]
fold_losses[f"fold_{number}_log_loss"] = log_loss(
    valid["label"], probability, labels=[0, 1]
)
# After the three folds:
score = fmean(fold_losses.values())
```

```bash
"${python4[@]}" -m acebet.train.historical_evaluation tune "$results" > "$m4/study.json"
```

## Recompute a loss and catch a reversed objective

Use the match IDs in M3's export to fit one trial again. The script computes log loss directly from the probability assigned to each observed outcome. Before running it, decide which end of the score table Optuna would choose if the direction were `maximize`.

```bash
"${python4[@]}" - "$results" "$comparison" "$m4/study.json" <<'PY'
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
from statistics import fmean
import sys

import numpy as np
import optuna
import pandas as pd

from acebet.dataprep.historical_results import read_results
from acebet.train.historical_evaluation import (
    FEATURES, development_results, logistic_pipeline, replay_frame,
)

results, comparison, study_path = map(Path, sys.argv[1:])
report = json.loads(study_path.read_text())
m3 = json.loads((comparison / "comparison.json").read_text())
for key in ("source_sha256", "data_sha256", "development_sha256"):
    assert report[key] == m3[key], key
membership_path = comparison / "membership.csv"
assert sha256(membership_path.read_bytes()).hexdigest() == report["fold_membership_sha256"]

membership = pd.read_csv(membership_path, dtype={"match_id": str})
frame = replay_frame(development_results(read_results(results))).set_index("match_id")
trials = report["trials"]
sample = trials[0]
scores = []
for number, fold in membership.groupby("split", sort=False):
    train = frame.loc[fold.loc[fold["partition"] == "train", "match_id"]]
    valid = frame.loc[fold.loc[fold["partition"] == "validation", "match_id"]]
    model = logistic_pipeline(sample["C"]).fit(train[FEATURES], train["label"])
    p = model.predict_proba(valid[FEATURES])[:, 1]
    loss = -np.log(np.where(valid["label"] == 1, p, 1 - p)).mean()
    assert isclose(loss, sample[f"{number}_log_loss"], abs_tol=1e-8, rel_tol=0)
    scores.append(loss)

assert isclose(fmean(scores), sample["mean_log_loss"], abs_tol=1e-8, rel_tol=0)
winner = min(trials, key=lambda row: (row["mean_log_loss"], row["trial_number"]))
assert winner["trial_number"] == report["best_trial_number"]

wrong = optuna.create_study(direction="maximize")
for row in trials:
    wrong.add_trial(optuna.trial.create_trial(value=row["mean_log_loss"]))
assert wrong.best_trial.number != winner["trial_number"]
print("Recomputed trial", sample["trial_number"], scores, fmean(scores))
print("Minimize selects", winner["trial_number"], winner["C"], winner["mean_log_loss"])
print("Maximize selects", wrong.best_trial.number, wrong.best_value)
PY
```

On the locked run, trial 11 selects `C ≈ 0.01656` with mean development loss `0.67979`. Reversing the direction selects trial 5 with loss `0.68689`. The winner improved on the untuned logistic loss of about `0.68092`. Compare it with the Elo row in the M3 report before deciding how much that improvement is worth. The winning score comes from the folds the search used to choose it.

Keep the M3 Parquet and fold membership with `$m4/study.json`. Carry the winning trial into [M4b](m4b.md), where you will record the fits and recover a candidate by run ID.
