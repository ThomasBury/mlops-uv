---
title: Tutorial
description: Train, promote, serve, and diagnose a tennis prediction in ten hands-on lessons.
hide:
  - toc
---

# From training data to a running prediction service { #tutorial }

Give a colleague a running tennis predictor. Show which model answered a
request and explain what you would inspect if the result looked wrong.

For example, here are three fields from a response for Fognini against Jarry.
The packaged default supplies this example. Your own export will have a
different version.

```json
{
  "p1_win_probability": 0.768387902579221,
  "model": {"version": "model_2026-08-02-12-59-07"},
  "feature_freshness": {"p1_as_of_date": "2018-03-04"}
}
```

- `p1_win_probability` gives Fognini about a 77% chance of winning.
- `model.version` identifies the saved model. The full response also includes
  a checksum, a fingerprint of the model file.
- `p1_as_of_date` dates the rank and Elo rating used for Fognini. Elo is a
  rating based on past match results. This example replays the known
  Brasil Open final using snapshots from the same match row on the last
  date in the dataset.

## Before you start

Install Git, Python 3.12, `uv`, `just`, and `curl`. Use a fresh checkout of
the [AceBet repository](https://github.com/ThomasBury/mlops-uv). The commands
assume a Unix-style shell and start from the repository root.
You will need a running Docker engine only for lesson 9 and the live
container check in lesson 10. Port 8000 needs to be free for the local API
and, later, the container.

## Read in order { #read-in-order }

| Lesson | Decision you can make when you finish |
| --- | --- |
| [1. Recreate the working environment](./00-recreate-the-working-environment.md) | Which command verifies that the checkout can read its training input? |
| [2. Establish what a prediction needs](./01-notebook-to-data-contract.md) | Is there enough player information to build the request's model inputs? |
| [3. Make a training decision reproducible](./02-feature-contract-training-and-export.md) | Which candidate should serve, given the recorded comparison? |
| [4. Choose the bundle the service may load](./03-reproducibility-versioning-and-dependencies.md) | Which saved files can the service load? |
| [5. Serve one prediction](./04-serving-async-and-monitoring.md) | Did this response come from your promoted bundle? |
| [6. Explain a rejected request](./05-authentication-authorization-and-access-boundaries.md) | What should the caller fix after a rejection? |
| [7. Investigate a running service](./06-observability-logging-metrics-and-traces.md) | What evidence answers questions about the request, its inputs, and forecast quality? |
| [8. Hand the application to another Python user](./08-hand-the-application-to-another-python-user.md) | What does a colleague receive when they install the wheel? |
| [9. Reproduce the service runtime](./09-reproduce-the-service-runtime.md) | Did Docker serve the same bundle as the local process? |
| [10. Decide what to verify before release](./07-validation-ci-deployment-and-boundaries.md) | Which check would settle a suspected model mix-up? |

## What this walkthrough establishes { #lifecycle }

You start with prepared historical data, compare models on one later period,
and finish with a locally tested service. Establishing the original data
source and how each input was available at prediction time takes more evidence.
Judging future forecasts takes predictions recorded before matches and their
later results. Operating a live deployment is separate work. See
[Tutorial Boundaries](../tutorial-shortcuts-vs-production.md).

## Where to look up details { #implementation-map }

Use [Training and Inference](../user-guide/training-and-inference.md) for the
shared inputs and evaluation, [Model Artifacts and Promotion](../user-guide/model-artifacts-and-promotion.md)
for the bundle and its validation, and [API Endpoints](../api/sync-endpoints.md)
for complete requests and responses. The optional
[starter notebook](../assets/notebooks/01-starter-feature-and-training-flow.ipynb)
uses a 100-row sample to show transformations and model comparison in memory.
