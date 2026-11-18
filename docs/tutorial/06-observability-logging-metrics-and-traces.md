---
title: Investigate a running service
description: Connect one request to its log, counter, input freshness, and the later evidence needed to score the forecast.
---

# 7. Investigate a running service { #tutorial-6-implemented-observability }

<a id="why-this-exists"></a>
<a id="exercise-the-surface"></a>

Imagine a colleague says the last prediction looked wrong. Collect the
evidence for one new request in terminal B while the local service in
terminal A is idle. Keep the same process running throughout because its
counter is stored in memory.

## Find this request in the logs and the counter { #experiment-count-one-production-prediction }

<a id="request-correlation-and-logs"></a>
<a id="bounded-metrics"></a>

```bash
curl -fsS "$api/metrics" | grep '^acebet_prediction_requests_total{mode="production"} '
token="$(just --quiet token)"
curl -fsS "$api/predict/production" \
  -H "Authorization: Bearer $token" \
  -H 'X-Request-ID: tutorial-freshness' \
  -H 'Content-Type: application/json' \
  --data-binary @var/tutorial/request.json \
  -D var/tutorial/observed-headers.txt \
  -o var/tutorial/observed-response.json
unset token
curl -fsS "$api/metrics" | grep '^acebet_prediction_requests_total{mode="production"} '
```

The counter should rise by one. To find the outcome of this particular
request, look for `X-Request-ID` in the saved headers and find the same ID
in terminal A's `http_request_completed` log. The status on that line tells
you how the API responded. The counter also includes failed scoring attempts.

## Decide what the response can tell you

Open `var/tutorial/observed-response.json`. Was the player information
current for the requested match? Did the forecast turn out well?

**Hint:** compare the dates under `feature_freshness` with the requested
match date. Think about what had to happen after a real forecast before you
could score it.

??? note "Check your explanation"

    In this replay, both snapshots are dated March 4, 2018, and their ages
    are zero days. They match the requested calendar date. The date alone
    gives no evidence that the information was available before the match.

    For one forecast, a later match result tells you the winner and lets
    you score the probability assigned to that outcome. To assess quality
    across cases, record predictions before matches with a match identifier,
    model version, and probability. Join them to the later outcomes and
    compute log loss over those matches, ideally alongside a baseline.

<a id="health-and-readiness"></a>
<a id="optional-tracing"></a>

See [Operational signals](../api/sync-endpoints.md#operational-signals) for
health and readiness, the full metrics list, and optional tracing.

**Finish when** you can point to the log status that shows the request
succeeded, the fields that date the inputs, and the later evidence you
would need to judge forecast quality. Stop the local server with **Ctrl-C
in terminal A**. Keep `request.json`, `local-response.json`, the model
bundle, and the snapshots for lessons 9 and 10. Continue with
[lesson 8](./08-hand-the-application-to-another-python-user.md).
