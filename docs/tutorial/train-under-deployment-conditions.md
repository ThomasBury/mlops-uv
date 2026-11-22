# 5. Train under deployment conditions

Suppose the service goes live on March 1. A player who competed in February is a perfectly reasonable March opponent. A February match whose result arrives after the March forecast cannot train that forecast. These two facts determine the split.

A **random split** asks how the model does on other rows drawn from the mixed archive. A **grouped split** withholds a whole group, useful for a question such as performance at unseen tournaments. Neither by itself enforces publication time. Here we want later matches, including returning players. Split by event day and admit a training label only after it becomes available. September onward stays closed during development.

<figure class="course-figure">
<div class="course-windows">
<div class="course-window"><span class="fit" style="--periods:1">Fit · Jan–Feb</span><span class="score">Validate · Mar–Apr</span></div>
<div class="course-window"><span class="fit" style="--periods:2">Fit · Jan–Apr</span><span class="score">Validate · May–Jun</span></div>
<div class="course-window"><span class="fit" style="--periods:3">Fit · Jan–Jun</span><span class="score">Validate · Jul–Aug</span></div>
</div>
<figcaption>Each row is a new fit. The training history expands. Labels must be visible by the end of the day before validation begins. Development labels also have to mature by August 31 at 23:59:59 UTC.</figcaption>
</figure>

## What the pipeline learns

Prevalence is the fraction of first-player wins in the fitting rows. Elo converts the pre-match rating gap directly into a probability. Logistic regression learns a weight on that gap and coefficients for the court surfaces. Its scaler learns the mean and spread of the training gaps. Its encoder learns which surfaces appeared there. In `_comparison`, the indices determine which rows supply both transformations.

```python
train, valid = frame.iloc[train_idx], frame.iloc[valid_idx]
model = logistic_pipeline().fit(train[FEATURES], train["label"])
```

Run the first fit yourself. Which match IDs and surfaces were available? If you fit the pipeline a second time on the entire development frame, which observations enter the scaler?

Save this as `inspect-fold.py` in your scratch directory.

```python
from pathlib import Path
import sys
from acebet.dataprep.historical_results import read_results
from acebet.train.historical_evaluation import (
    FEATURES, development_folds, development_results, logistic_pipeline, replay_frame,
)

frame = replay_frame(development_results(read_results(Path(sys.argv[1]))))
train_idx, valid_idx = development_folds(frame)[0]
train, valid = frame.iloc[train_idx], frame.iloc[valid_idx]
print(train[["match_id", "available_at"]].tail(), valid[["match_id", "cutoff"]].head())
assert train.available_at.max() <= valid.cutoff.min()
assert set(train.match_id).isdisjoint(valid.match_id)
for rows in (train, frame):
    fitted = logistic_pipeline().fit(rows[FEATURES], rows.label)
    transforms = fitted.named_steps["features"].named_transformers_
    print(len(rows), round(transforms["elo"].mean_[0], 3), transforms["surface"].categories_[0])
print(frame.loc[~frame.index.isin(train_idx), ["match_id", "event_date", "available_at"]])
```

```bash
.venv/bin/python "$work/inspect-fold.py" "$work/matches.parquet"
```

The first fit uses 234 matches and sees Clay and Hard. Its mean is **2.694** Elo points. The full development fit also sees Grass. Fitting on all development rows gives **4.152**. The extra rows include later validation matches and any earlier labels that missed the first fit’s deadline. A scaler can learn about future opponents without ever reading their labels. Reuse the first fit’s scaler and encoder on its validation window. [scikit-learn pipeline and leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)

## Compare on the same fixtures

```bash
.venv/bin/python -m acebet.train.historical_evaluation compare "$work/matches.parquet" > "$work/comparison.json"
cat "$work/comparison.json"
```

Mean fold log loss gives equal weight to each validation window. Prevalence scores 0.69381, Elo 0.67036, default logistic regression 0.68092. Feeding the current winner into Elo yields 0.58619. Under `random_split_diagnostic`, the latest training label arrives after all 444 validation cutoffs. Its loss of 0.66743 describes a different set of fixtures. The timestamps expose the error.

The archive supplies assumed result delays and completed results only. These folds model future rows under that assumption. Next, [freeze one candidate](./evaluate-a-forecast.md) before consulting the final matches.
