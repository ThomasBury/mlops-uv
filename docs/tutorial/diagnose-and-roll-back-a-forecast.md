# 9. Investigate a forecast after serving

A request took longer than usual. More requests are coming from hard courts. Which observation would tell you the model needs attention? First follow one request from arrival to reply. Then ask which of those forecasts you can actually score.

## Locate the slow part

Start a monitored worker using your original assets. JSON logs record individual forecasts. A Prometheus scrape summarizes activity, and OpenTelemetry spans time operations within a request. Each worker has its own counters.

```bash
base=http://127.0.0.1:8012
ACEBET_FORECAST_RESULTS="$work/matches.parquet" ACEBET_FORECAST_BUNDLE="$work/first" \
  ACEBET_LOG_LEVEL=INFO ACEBET_LOG_FORMAT=json ACEBET_LOG_FILE="$work/requests.jsonl" \
  ACEBET_TRACING_ENABLED=true ACEBET_TRACE_EXPORTER=console ACEBET_METRICS_ENABLED=true \
  .venv/bin/uvicorn acebet.app.historical:create_app --factory --host 127.0.0.1 --port 8012 \
  > "$work/spans.txt" 2>&1 &
monitor_pid=$!
curl -fsS --retry 12 --retry-connrefused "$base/readyz"
curl -fsS "$base/metrics" > "$work/before-metrics.txt"
for id in F01 424816; do
  curl -fsS -D "$work/$id-headers.txt" -H 'Content-Type: application/json' \
    --data-binary @"$work/$id-request.json" "$base/forecast" > "$work/$id-monitored.json"
done
curl -fsS "$base/metrics" > "$work/after-metrics.txt"
```

Find `X-Request-ID` in the archived fixture’s response headers. Find that ID in `requests.jsonl`. The `http_request_completed` event gives status and duration. The `forecast_served` event records the model, state and returned probability. Find its `trace_id` in `spans.txt` after the console exporter flushes. Compare the durations of `forecast.feature_lookup` and `forecast.inference` under the same request.

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

Inject a controlled delay using the existing monitoring test. It holds one inference call while probing the gauge, releases it and then sends four hard-court requests. Which span grew? Did a rejected input add a probability?

```bash
.venv/bin/pytest -q -s tests/test_historical_serving.py::test_monitoring_signals
```

The injected delay adds at least 300 ms to inference. A 422 increases the error count while the number of successful predictions stays fixed. A change in the surface mix is input drift. A shift in returned probabilities is prediction drift. Both can occur with a healthy model.

## Advance the reporting clock

Fixture 424816 was played on September 20. Under the archive’s two-day availability assumption its winner becomes visible at midnight UTC on September 22. Run the report twice on that log, one second apart on the reporting clock.

```bash
for clock in 2021-09-21T23:59:59Z 2021-09-22T00:00:00Z; do
  .venv/bin/python -m acebet.historical_monitoring quality \
    "$work/requests.jsonl" "$work/matches.parquet" "$clock"
done
kill "$monitor_pid"
```

The function `quality_report` admits a result when `match.available_at <= as_of`. The denominator is the number of distinct recorded forecasts in the cutoff month for a model and history. Retries and reversed player order count once. Before midnight, September has one forecast, zero scored labels and null loss. At midnight it still has one forecast, now with a loss and one calibration-bin observation. F01 belongs to 2030 and contributes to neither report.

**Concept drift** means the relationship between inputs and outcomes changes. Investigate it if enough mature labels show a sustained quality change among comparable fixtures. With one label, check the join, order and coverage first. Missing logs or selective label delays can bias that comparison. Carry the evidence into the [maintenance decision](./automate-and-audit-a-forecast.md).
