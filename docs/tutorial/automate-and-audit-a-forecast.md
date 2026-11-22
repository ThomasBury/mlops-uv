# 10. Decide what a scheduled run may change

A candidate build finished and its mean loss is slightly lower than Elo’s. Before giving a schedule permission to change anything, make the decision yourself. Which end of the paired interval must be below zero? What exactly would you have to restore if this service began returning different forecasts?

## Apply the release rule

The rule declared before final evaluation requires the upper end of the weekly 95% interval for candidate-minus-Elo loss to be below zero. Open `first/final/evaluation.json` and write down your decision. Check your decision against the row-level predictions.

```bash
.venv/bin/python -m acebet.historical_monitoring gate "$work/first/final" \
  --candidate logistic_tuned --incumbent elo
```

Expect exit status 1. The upper endpoint is about +0.00057. A positive value allows the candidate to be worse. Keep the recorded model available for replay. Reusing these same final fixtures to choose another setting would turn them into selection data.

## Restore the model and its history

Run the recovery drill. During the test, an alternate model is trained on history with one simulated late result. A mixed pair must fail readiness. Restoring the original model **and** history must recover both original replies. Which fixture would reveal a change to the archived feature state?

```bash
.venv/bin/pytest -q -s tests/test_historical_serving.py::test_gate_and_rollback
```

Read the two assertions on the restored replies at the end of the test. They compare every field with the original response. F01 uses synthetic history. The archived fixture actually looks up ratings in the restored results.

## Automate the commands you can already explain

Install the `just` command runner. A recipe gives a shell workflow a short name. In `justfile`, follow the calls inside `forecast-candidate`. The recipe creates a fresh run directory, prepares matches, trains the candidate, evaluates it and writes a review. Use a new path on a retry so you can inspect any interrupted run.

```bash
run="$work/candidate"
just forecast-test
just forecast-candidate "$run"
cat "$run/validation.json"
```

Trace `validation.json` back to the evidence. Its fixture records the model and history digests, its evaluation digest identifies the scored run, and its gate records both the interval and `accepted: false`. The review command succeeds when it has produced valid evidence, including a rejection. Try to promote that run explicitly.

```bash
just forecast-promote "$run"
```

Expect exit status 1. A rejection leaves the run directory without a promotion record. Check it directly.

```bash
test ! -e "$run/bundle/promotion.json"
```

A passing explicit promotion writes a dated approval. Starting a worker with a particular model and history is a separate operator action. The historical GitHub Actions workflow runs the tests and candidate recipe on pull requests, pushes to main, manual dispatch and a Monday schedule. It uploads the evidence with read-only repository permissions. Replaying the pinned snapshot can detect a reproducibility regression. A new model-value decision needs fresh, mature evidence. [CI workflow reference](../cicd/github-actions-pipeline.md)

## Diagnose an unfamiliar forecast

Ask a facilitator for a prepared checkout and an unfamiliar forecast. Identify the model, feature state, player order and cutoff. Choose a claim from the [coverage matrix](./index.md) and design an independent check small enough to explain by hand. Record the failing observation, repair the cause and demonstrate the same check passing. Give the fixture ID, revision and the evidence that led you there.

The ten-minute orientation and this blind diagnosis are reader acceptance checks. Both await a human walkthrough.
