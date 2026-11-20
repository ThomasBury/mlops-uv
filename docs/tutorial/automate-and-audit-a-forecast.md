---
title: Automate and audit a forecast
description: Rebuild a candidate from a locked checkout, review its evidence, and repair an undisclosed defect.
---

# Automate and audit a forecast

A scheduled run produces a plausible probability for F01. Before approving
the candidate, find out which model and history produced it, whether the
comparison cleared the gate, and how someone else could challenge the result.
Use the evidence from the [M1–M6 exercises](./index.md) as your reference.

## Run the workflow from a clean checkout

Use Python 3.12, `uv`, and `just` from the checkout root. The recipe uses a
new run directory so an interrupted run cannot be mistaken for a fresh one.
Keep a failed directory for inspection and use a new path for the retry.

```bash
uv sync --locked --python 3.12 --group mlops --extra tracing
just forecast-test
work="$(mktemp -d)"
run="$work/candidate"
just forecast-candidate "$run"
```

`forecast-candidate` calls `forecast-prepare`, `forecast-train`, and
`forecast-evaluate` in that order. Run `just --list` to see their arguments.
Start `just forecast-serve "$run/matches.parquet" "$run/bundle" 8010`.
From another shell, send the F01 request and compare the response with
`fixture` in `validation.json`. Stop the server after the comparison.

```bash
curl -fsS http://127.0.0.1:8010/readyz
curl -fsS -H 'Content-Type: application/json' \
  --data-binary '{"fixture_id":"F01","p1_id":"P01","p2_id":"P02","surface":"Clay","cutoff":"2030-06-10T12:00:00Z"}' \
  http://127.0.0.1:8010/forecast
```

| Evidence in the run directory | What to check |
| --- | --- |
| `preparation.json`, `matches.parquet` | Source and canonical content hashes, 1,856 completed matches. |
| `bundle/selection.json`, `bundle/membership.csv`, `bundle/search.csv` | Frozen model hash, training and validation dates, fitted search setting, and F01 feature trace. |
| `bundle/final/evaluation.json`, `bundle/final/forecasts.csv` | 369 final forecasts, paired losses and calibration. |
| `validation.json` | F01 probability near `0.46910`, model and history digests, evaluation digest, and the Elo gate decision. |

The **Historical forecast** GitHub Actions workflow runs the same checks and
replays the candidate on pull requests and pushes to `main`. Once the workflow
is on the default branch, it can run on demand and runs on Mondays at 06:23 UTC.
Open a run's **historical-forecast** artifact to inspect the validation record
and its supporting files. Each run uses the pinned 2021 snapshot. Repeating it
checks reproducibility. A new promotion decision for a live system needs a
fresh, mature evaluation period and a rule fixed in advance.
[GitHub Actions schedule and manual triggers](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)

## Make an explicit promotion decision

Use the run you built locally. Loading a Joblib bundle requires trusting who
produced it. M4 declared that the candidate must beat Elo across the paired 95%
weekly interval. The archive's candidate-minus-Elo interval is approximately
`[-0.00250, 0.00057]`. The upper end is above zero, so expect `accepted: false`
in the review. A successful scheduled job means it produced valid evidence.

```bash
just forecast-promote "$run"
```

The promotion command reevaluates the gate from the recorded forecasts and
checks that the evaluation matches the model and selection. Expect exit
status 1 for this archive and check that `"$run/bundle/promotion.json"` is
absent. When the gate passes, this explicit command writes a dated approval
with the evaluation, model, and history identities. The regression test
exercises that path with synthetic scores. Starting a service with a chosen
bundle and history remains a separate operator action. Rehearse recovery with
the [M6 rollback drill](./diagnose-and-roll-back-a-forecast.md#reject-a-candidate-then-rehearse-rollback).

A local record is enough for one operator. Consider a registry when approvals
must be shared across environments, an orchestrator when retries and backfills
span dependent jobs, and a feature store when multiple systems need consistent
historical and online features.

## Find the verification evidence

Use `just forecast-test` for the named historical tests below, or run a single
test after the locked sync with `.venv/bin/pytest -q PATH::TEST`.

| Claim | Command or exercise | Passing observation and evidence |
| --- | --- | --- |
| Defined target | [Classify the worked fixtures](./forecast-contract.md#try-the-contract-on-five-fixtures) before reading the answer key. | Fixed player order allows both labels. Recover the baseline loss on the two scorable fixtures. |
| Reproducible environment | [Install and replay outside the checkout](./reproduce-the-fixture.md#replay-and-verify-f01). Run `.venv/bin/ruff check src tests` and `.venv/bin/pytest tests` after the locked sync. | The installed command gives `F01` and `0.50`. Keep its JSON and the check output. |
| Auditable data | [Verify and rebuild the snapshot](./rebuild-match-history.md#verify-the-source-before-using-it). | Match the revision, source SHA-256, canonical schema and content digest. Inspect the Parquet rows and `preparation.json`. |
| Idempotent ingestion | `test_historical_pipeline.py::test_rebuild_preserves_source_and_reproduces_canonical_content` | Rebuilding to the same path preserves identities, row count, labels, and digest. |
| No future information | `test_historical_pipeline.py::test_cutoff_ignores_late_corrections_future_matches_and_current_winner` and the [leaky comparison](./rebuild-match-history.md#measure-the-effect-of-using-the-current-winner). | Earlier forecasts survive the injected changes. The leaky score uses the current outcome and appears better. |
| Deployment-like validation | `test_historical_evaluation.py::test_selection_excludes_late_labels_and_fits_preprocessing_on_each_training_window` | Inspect `membership.csv` for disjoint match IDs within each split, mature training labels, and returning players. `final_refit` excludes the final period. |
| Measured model value | [Evaluate the final period](./evaluate-a-forecast.md#open-the-final-period). | `final/evaluation.json` reports common-fixture loss, calibration, and paired uncertainty. The interval includes zero against Elo. |
| Reproducible training | [Rebuild, retrain, and compare](./evaluate-a-forecast.md#rebuild-and-reproduce-the-evidence). | Match identities and settings exactly, replay probabilities and scores within `1e-8`. |
| Training and serving agree | `test_historical_serving.py::test_frozen_replay_matches_batch_and_concurrent_http` | Match F01 and a held-out replay to the frozen records, including model and feature-state identities. |
| Reliable deployment | [Build the image and test readiness](./serve-the-frozen-forecast.md#put-the-assets-inside-the-image). | Keep the image ID, non-root UID, zero mounts, matching and concurrent replies, 422 for invalid input, and 503 readiness with corrupt assets. |
| Monitoring signals | [Inject faults and withhold labels](./diagnose-and-roll-back-a-forecast.md). | Inspect the error count, timing and input mix, request trace, and coverage before and after the archived result arrives. |
| Rollback | `test_historical_serving.py::test_gate_and_rollback` and `test_historical_serving.py::test_candidate_review_and_explicit_promotion` | A rejection produces no approval. A mismatched deployment is unready. Restored history and model reproduce both fixture replies. |

## Run a blind walkthrough

**Learner brief.** You have a working-looking pipeline and an unfamiliar
forecast. Identify the model, the feature state, and the cutoff. Choose one
claim from the table that the result should satisfy. Show a counterexample
with a small input, explain which claim fails, repair the cause, and supply a
regression check that fails on the original copy and passes after the repair.

**Facilitator setup.** In a disposable copy, find `results_as_of` in
`src/acebet/dataprep/historical_replay.py` and replace the check
`result.available_at <= cutoff` with `result.event_date <= cutoff.date()`.
Build a new candidate with `just forecast-candidate` and find fixture `425753`
in `bundle/final/forecasts.csv`. Its Elo gap changes from about `-127.35` to
`-123.80` while it still has a plausible probability. Give the learner the
modified copy, that forecast, and the learner brief. Withhold the original
results, the change, and its location. A diagnosis should show why a result
from an earlier event can arrive after the cutoff and why its label must be
excluded. Use the cutoff regression in the table to verify the repair.

**Walkthrough record.** Capture the date, checkout revision, fixture ID,
the learner's affected claim, the failing command and observation, the repair
diff, and the passing regression output. Ask the learner which evidence led
to the cause. Record that independently of CI. Status is pending until a
human learner completes the walkthrough.
