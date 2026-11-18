---
title: Make a training decision reproducible
description: Train once, compare candidates on later matches, and retain the selected bundle and its decision record.
---

# 3. Make a training decision reproducible { #tutorial-2-model-validity-selection-and-the-feature-contract }

<a id="why-this-exists"></a>

Choose which candidate should answer the request you saw at the start. Use a
**baseline**, a simple method that sets a minimum standard, so a more complex
model has to earn its place.

## Run the comparison and export once { #run-the-comparison-without-exporting }

Start with an unused `var/tutorial/models/` directory so the export is the
only bundle available for the next lesson.

```bash
just train-local var/tutorial/models
cat var/tutorial/models/model_*.modelcard.md
```

Training compares three candidates. Constant predicts the same probability
for every match using the fraction of player-1 wins in the fitting data. Elo
uses the two ratings. LightGBM learns from the match inputs. The command exports
the selected model and two companion files. The **model card** summarizes
the comparison for a reader. The `.metadata.json` file records exact values
for the program.

## Check which matches informed the decision { #one-strict-date-grouped-selection-holdout }

Open the `.metadata.json` file. Under `dataset`, find `train_date_range`,
`test_date_range`, and `row_counts`. The **selection holdout** is the later
set of matches used to choose among candidates. The `test` keys name that
set.

| Use | Rows | Date range |
| --- | ---: | --- |
| Fit candidates | 3,482 | 2015-03-12 through 2016-06-19 |
| Compare candidates | 1,754 | 2016-06-20 through 2017-03-04 |

Keeping an entire date on one side of the split makes the fitting period
strictly earlier. This later period informs the selection decision. An
independent estimate of forecast quality would use outcomes that played no
part in choosing the model.

<a id="prediction-time-availability-comes-first"></a>
<a id="preprocessing-respects-feature-type"></a>

A **feature** is an input to a model. The shared feature list allows match
context and player values such as rank and Elo. Letting a final score in
would be **leakage**, because that value arrives after the forecast is needed.
The same feature list applies when the service scores a request. See
[Training and Inference](../user-guide/training-and-inference.md) for the
full list, preprocessing, and limits of the historical inputs.

## Read the evidence for the selection { #canonical-candidate-evidence }

<a id="probability-quality-and-calibration-limits"></a>

**Log loss** scores the probability assigned to what actually happened.
Lower is better, and confident mistakes cost more. Here are the results
recorded with the packaged model for this comparison.

| Candidate | Accuracy | Log loss | Brier score | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| Constant | 0.5006 | 0.6931 | 0.2500 | 0.5000 |
| Elo | 0.6762 | 0.6021 | 0.2066 | 0.7410 |
| LightGBM | 0.6631 | 0.6181 | 0.2138 | 0.7239 |

Use log loss for the decision. Either servable candidate must improve
on constant by at least `0.005`. To replace a qualifying Elo model, LightGBM
must improve on Elo by at least `0.005`. The policy compares exact values
from metadata.

<a id="experiment-change-hypothetical-candidate-evidence"></a>

**Finish when** you can point to the fitting and comparison dates and explain
the recorded decision. The canonical metadata records `elo_baseline` with
reason `elo_preferred_challenger_margin_not_met`. What would the policy
choose if LightGBM's log loss were `0.5960` and Elo's stayed at `0.6021`?
Keep all three exported files for [lesson 4](./03-reproducibility-versioning-and-dependencies.md).
