---
title: Diagnose and roll back a forecast
description: Trace a slow forecast, score it when its result arrives, and restore a known-good deployment.
---

# Diagnose and roll back a forecast

A request is slow, and the service starts seeing more hard-court matches.
Which forecasts were affected? Which results have arrived? Use the service
from [M5](./serve-the-frozen-forecast.md), including its `$work/first`
bundle, `$work/matches.parquet` and `$work/f01.json`. Run from the
checkout root with the same locked Python environment.

## Identify the request and the work it did

Start one worker with JSON logging and the local OpenTelemetry console exporter.
Use a fresh log file for this run. In the server shell:

```bash
ACEBET_FORECAST_RESULTS="$work/matches.parquet" \
ACEBET_FORECAST_BUNDLE="$work/first" \
ACEBET_LOG_LEVEL=INFO \
ACEBET_LOG_FORMAT=json \
ACEBET_LOG_FILE="$work/m6-requests.jsonl" \
ACEBET_TRACING_ENABLED=true \
ACEBET_TRACE_EXPORTER=console \
ACEBET_METRICS_ENABLED=true \
uv run --locked --group mlops --extra tracing uvicorn acebet.app.historical:create_app \
  --factory --host 127.0.0.1 --port 8012 > "$work/m6-spans.txt"
```

In a second shell, set `work` to the same directory and send F01.

```bash
base=http://127.0.0.1:8012
curl -fsS "$base/readyz"
curl -fsS -D "$work/m6-f01.headers" -H 'Content-Type: application/json' \
  --data-binary @"$work/f01.json" "$base/forecast" > "$work/m6-f01.json"
curl -sS -w 'HTTP %{http_code}\n' -o "$work/m6-invalid.json" \
  -H 'Content-Type: application/json' -d '{}' "$base/forecast"
curl -fsS "$base/metrics" > "$work/m6-metrics.txt"
```

Expect readiness, a forecast matching M5, and a 422 for the empty request.
Find the `X-Request-ID` response header in the JSON request log. The
`http_request_completed` event records the status and duration. For a
successful forecast, `forecast_served` also records the forecast ID, cutoff,
model and feature state, and returned probability. Find that event's
`trace_id` in `m6-spans.txt`. The exporter batches spans, so allow a few
seconds. The same trace contains the HTTP server span and the two child
spans `forecast.feature_lookup` and `forecast.inference`. Their timings
show where the server spent time.

Inspect the Prometheus scrape before and after a batch of requests.

| What to check | Signal |
| --- | --- |
| Throughput | Change in `acebet_http_requests_total` for `POST /forecast` divided by elapsed time. |
| Errors | Share of those requests with a 4xx or 5xx status. Look up the status and request ID in the logs. |
| Latency | Changes in the buckets of `acebet_http_request_duration_seconds`. A bucket counts requests at or below its upper bound. |
| Concurrency | `acebet_forecasts_in_progress` sampled while forecast handlers are running. |
| Resource use | On Linux, `process_resident_memory_bytes` and the change in `process_cpu_seconds_total`. |
| Input mix | `acebet_forecast_surface_total` and the absolute Elo gap histogram. |
| Forecast mix | `acebet_forecast_p1_win_probability` histogram. Keep player order consistent when comparing batches. |

Logs and traces identify requests. Metric labels stay limited to values
such as route and surface so that each new fixture does not add a time
series. Here the metrics describe one process and reset on restart.

Run the controlled faults against the same application in a test process.

```bash
uv run --locked --group mlops --extra tracing pytest -q -s \
  tests/test_historical_serving.py::test_monitoring_signals
```

The invalid request adds one 422 and no probability observation. The slow
inference holds the in-progress gauge at one, takes more than 300 ms in the
inference span, and adds a completed request above the 250 ms bucket. Four
subsequent hard-court requests increase the Hard counter while the Clay
counter stays fixed. Compare the probability distributions separately. A
different input mix can change predicted probabilities even when the model
and its probability quality are stable.

## Score the forecast as labels arrive

Replay hard-court fixture `424816`. It uses the same model as F01 and has a
recorded result. The dates in the commands below advance a simulated
reporting clock through the result's availability time. They use the
archive's [availability rule](./rebuild-match-history.md).

```bash
uv run --locked --group mlops --extra tracing python - "$work" <<'PY'
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

work = Path(sys.argv[1])
with (work / "first/final/forecasts.csv").open() as stream:
    row = next(row for row in csv.DictReader(stream) if row["match_id"] == "424816")
(work / "m6-hard.json").write_text(json.dumps({
    "fixture_id": row["match_id"],
    "p1_id": row["p1_id"],
    "p2_id": row["p2_id"],
    "surface": row["surface"],
    "cutoff": datetime.fromisoformat(row["cutoff"]).isoformat(),
}))
available = datetime.fromisoformat(row["available_at"])
(work / "m6-before-label.txt").write_text((available - timedelta(seconds=1)).isoformat())
(work / "m6-label-available.txt").write_text(available.isoformat())
PY
curl -fsS -H 'Content-Type: application/json' \
  --data-binary @"$work/m6-hard.json" "$base/forecast"
for clock in m6-before-label m6-label-available; do
  uv run --locked --group mlops --extra tracing python -m acebet.historical_monitoring \
    quality "$work/m6-requests.jsonl" "$work/matches.parquet" "$(cat "$work/$clock.txt")"
done
```

The September cohort has one forecast in both reports. Just before the
result becomes available, coverage is zero, and log loss and calibration
are `null`. Once the result is visible, coverage is one and both metrics
can be computed. One observation gives no reliable estimate of overall
calibration. F01's 2030 cutoff is outside these reporting windows and it
has no result in the archive.

The report uses recorded probabilities and joins fixture IDs to compatible
results visible by the reporting time. Retries and swapped player order
count once for a model and feature state. Calibration groups forecasts by
probability and compares the mean prediction with the observed win rate.
The denominator includes pending forecasts within each cutoff month and
deployment. If late labels are concentrated on particular fixtures, the
currently labeled subset can give a misleading picture of the whole
period. Throughput, latency and input mix are observable while results
are pending. A change in forecast quality needs mature, representative
labels and enough fixtures to put the change in context.

This replay uses a local rotating request log. Complete coverage depends
on retaining the full log for the period. Use the log of this short run
without rotation. In a longer run, collect all rotations or persist
forecast events before using them as the denominator.

## Reject a candidate, then rehearse rollback

Use the saved M4 evaluation to rehearse an acceptance decision. The
candidate is the constant training prevalence score, and the incumbent
is the M5 logistic model. Both are evaluated on the same 369 fixtures.
The gate accepts only when the upper end of the paired 95% weekly
bootstrap interval for candidate-minus-incumbent log loss is below zero.

```bash
uv run --locked --group mlops --extra tracing python -m acebet.historical_monitoring \
  gate "$work/first/final" --candidate prevalence --incumbent logistic_tuned
```

Expect `"accepted": false` and exit status 1. The measured difference is
about +0.0682 log loss, with a paired interval of +0.0392 to +0.0877. Replay
F01 after the gate and verify the same response as before. This gate
reads recorded evidence and returns a decision. Activation is a separate,
explicit operation. An actual new candidate needs a fresh evaluation
period and a declared acceptance rule before looking at the results.

Next, rehearse recovery from an already deployed change in an isolated
service lifecycle.

```bash
uv run --locked --group mlops --extra tracing pytest -q -s \
  tests/test_historical_serving.py::test_gate_and_rollback
```

The drill trains a second model from a copy of the snapshot with one
simulated late result, then starts the service with that model and history.
It verifies different checksums and confirms that a mixed pair stays
unready. The rollback starts the service with both original assets and
replays F01 and the archived fixture. Both responses agree with the
originals, including the probability, model digest and feature-state
digest. F01 uses its synthetic trace, so the archived fixture verifies the
historical feature-state change.

Keep the original and alternate model and history identities, the gate
decision, and the before-and-after responses for M7. Stop the local server
when finished.
