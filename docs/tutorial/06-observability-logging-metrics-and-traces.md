---
title: Tutorial 6 - Implemented Observability
description: Inspect request IDs, bounded metrics, optional tracing, and minimal liveness and readiness behavior.
---

# Tutorial 6: Implemented Observability

AceBet implements a small in-process observability surface. This chapter covers
only those signals; it does not imply a centralized operations platform.

## Why This Exists

**Observability** is the evidence needed to understand a running service. The
commands below prove requests can be correlated, counted, and checked for
readiness without claiming a centralized operations platform.

## Exercise The Surface

With the API running:

```bash
just healthz
just readyz
just metrics
just predict-production
```

## Request Correlation And Logs

Middleware preserves an incoming `X-Request-ID` or creates one, echoes it on
the response, and binds it into structured request logs. Logs include method,
path, status, and duration. Request bodies are not logged by default. Context is
cleared in a `finally` path so one request cannot leak into the next.

## Bounded Metrics

`/metrics` exposes Prometheus text for HTTP counts and duration, prediction
counts and duration, prediction failures, and successful sample-data use.
Known endpoints use route templates as labels. Unknown URLs collapse to the
fixed `__unmatched__` label, and `/metrics` excludes itself from generic HTTP
metrics. This prevents arbitrary request paths from producing unbounded label
values. The nine standard HTTP methods keep their names; every other method
uses `OTHER` in both counters and histograms.

## Optional Tracing

Tracing is disabled unless the `tracing` extra and
`ACEBET_TRACING_ENABLED=true` are supplied. FastAPI request spans and a
prediction span can be exported to the console or OTLP. Shutdown un-instruments
the app, flushes and shuts down the app-owned provider, and clears tracing
state. FastAPI's native telemetry is explicitly disabled, including automatic
exporter configuration, so it cannot create a second telemetry lifecycle.

## Health And Readiness

`/healthz` returns `{"status":"ok"}` when the process can respond.
`/readyz` returns ready only when the runtime assets required for the current
environment were loaded during lifespan startup. These checks are local and
in-process; they do not probe an external registry, data store, collector, or
downstream dependency.

There are no bundled dashboards, alerts, log shipping, retention rules, SLOs,
or centralized trace backend. See the single
[Tutorial Boundaries](../tutorial-shortcuts-vs-production.md) matrix.

!!! info "Shortcut"

    **Why it is enough here:** local logs, bounded metrics, and optional traces
    make one process inspectable. **Add it when…** operations need retention,
    alerts, SLOs, or shared dashboards; then add managed backends. See
    [Tutorial Boundaries](../tutorial-shortcuts-vs-production.md).

## Experiment: Count One Production Prediction

**Prerequisites:** run `uv sync --locked` and keep one otherwise idle local API
running on `127.0.0.1:8000`. Use the repository root, default packaged assets,
and configured demo credentials. Do not restart the API or send other
predictions during the experiment.
**Prediction:** scraping before and after one successful production prediction
shows a counter increase of exactly one.

```bash
uv run python - <<'PYCODE'
import subprocess
from urllib.request import urlopen


def prediction_count():
    with urlopen("http://127.0.0.1:8000/metrics") as response:
        metrics = response.read().decode()
    prefix = 'acebet_prediction_requests_total{mode="production"} '
    return next((float(line[len(prefix):]) for line in metrics.splitlines()
                 if line.startswith(prefix)), 0.0)


before = prediction_count()
subprocess.run(["just", "--quiet", "predict-production"], check=True)
after = prediction_count()
print(f"Prediction counter change: {after - before:.0f}")
assert after == before + 1
PYCODE
```

Expected checkpoint: the canonical successful prediction JSON, then
`Prediction counter change: 1`. The starting count depends on previous requests.

**Explain:** why is the difference useful even when the initial count varies?
Why must the API stay running, and why must other predictions be excluded?
**Cleanup:** no files change; the in-process counter resets when you stop the API
with Ctrl-C.

Continue with
[Validation And Release Boundaries](./07-validation-ci-deployment-and-boundaries.md).
