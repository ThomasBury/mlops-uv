---
title: Serve the frozen forecast
description: Replay the same forecast offline, over HTTP, and from a non-root container.
---

# Serve the frozen forecast

Use the candidate frozen in [the model comparison](./evaluate-a-forecast.md).
Check that packaging preserves the model, feature state, and probability.
Keep `$work/matches.parquet` and `$work/first` from that lesson and run these
commands from the checkout root.

## Replay one fixture locally

F01 compares Alex Silva (`P01`) with Robin Novak (`P02`) on clay at noon UTC.
It uses the [worked synthetic history](./rebuild-match-history.md). The
service also replays fixtures from `first/final/forecasts.csv` using the
recorded players, surface, and prior-day cutoff. For those fixtures, use
`match_id` as `fixture_id` and replace the space in the CSV timestamp with
`T`. A live service would need a trusted schedule and results feed available
at issue time.

```bash
cat > "$work/f01.json" <<'JSON'
{
  "fixture_id": "F01",
  "p1_id": "P01",
  "p2_id": "P02",
  "surface": "Clay",
  "cutoff": "2030-06-10T12:00:00Z"
}
JSON
uv run --locked --group mlops python -m acebet.historical_forecast \
  "$work/matches.parquet" "$work/first" < "$work/f01.json" > "$work/offline.json"
cat "$work/offline.json"
```

Expect `p1_win_probability` near `0.46910` and `elo_gap` of `-16`. The gap is
the first player's Elo minus the second player's Elo. The pipeline evaluates
the players in stable ID order. Reverse the request order and the gap
changes sign and the forecast becomes `1 - p`.

`model_sha256` identifies the fitted pipeline, including preprocessing.
`history_sha256` identifies the canonical history recorded by M4.
`feature_state_sha256` identifies the F01 trace or the canonical history
used for a final-period fixture. If the probability looks wrong, compare
the player order, cutoff, and these checksums before inspecting the Elo gap.

## Make the same forecast over HTTP

The FastAPI startup hook verifies the data, feature contract, feature source
files, and dependency versions. It checks the model bytes before loading and
then compares the F01 trace and probability with the selection record.
Serve bundles you trust because Joblib deserialization can execute code.
Python patch versions may differ between the local and container runtimes.
The recorded inference libraries must match exactly.
[FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/)

From one terminal, start Uvicorn with a single worker.

```bash
ACEBET_FORECAST_RESULTS="$work/matches.parquet" \
ACEBET_FORECAST_BUNDLE="$work/first" \
uv run --locked --group mlops uvicorn acebet.app.historical:create_app \
  --factory --host 127.0.0.1 --port 8010 --workers 1
```

In a second terminal, set `work` to the same directory and send the request.

```bash
curl -fsS http://127.0.0.1:8010/healthz
curl -fsS http://127.0.0.1:8010/readyz
curl -fsS -H 'Content-Type: application/json' \
  --data-binary @"$work/f01.json" http://127.0.0.1:8010/forecast > "$work/http.json"
```

Expect HTTP 200 from each command. With `{}` as the request body, Pydantic
returns HTTP 422 for missing fields. Moving the cutoff to 12:30 UTC also
returns 422 because the request no longer matches F01.

## Put the assets inside the image

Give Docker a build context containing the three serving assets. The
`historical-forecast` target uses the repository's multi-stage build and
runs the installed package as `appuser`.
[Docker multi-stage builds](https://docs.docker.com/build/building/multi-stage/)

```bash
mkdir -p "$work/image-assets"
install -m 0644 "$work/matches.parquet" "$work/first/selection.json" \
  "$work/first/model.joblib" "$work/image-assets/"
docker buildx build --target historical-forecast \
  --build-context forecast_assets="$work/image-assets" --load -t acebet:m5 .
image_id="$(docker image inspect acebet:m5 --format '{{.Id}}')"
printf '%s\n' "$image_id"
container="$(docker run --rm -d -p 127.0.0.1:8011:8000 "$image_id")"
docker exec "$container" id -u
docker inspect "$container" --format '{{len .Mounts}}'
curl -fsS --retry 10 --retry-connrefused http://127.0.0.1:8011/readyz
curl -fsS -H 'Content-Type: application/json' \
  --data-binary @"$work/f01.json" http://127.0.0.1:8011/forecast > "$work/container.json"
```

Expect a nonzero user ID and zero mounts. Record the `sha256:…` image ID to
identify the local build. A registry reports a manifest digest after
publishing. Compare all three responses, then check concurrent requests.

```bash
uv run --locked --group mlops python - "$work" <<'PY'
from concurrent.futures import ThreadPoolExecutor
import json
from math import isclose
from pathlib import Path
import sys

import httpx

work = Path(sys.argv[1])
offline, local, image = [
    json.loads((work / name).read_text())
    for name in ("offline.json", "http.json", "container.json")
]
probability = "p1_win_probability"
for replay in (local, image):
    assert offline.keys() == replay.keys()
    assert all(offline[key] == replay[key] for key in offline if key != probability)
    assert isclose(offline[probability], replay[probability], abs_tol=1e-8, rel_tol=0)
payload = json.loads((work / "f01.json").read_text())
with httpx.Client(base_url="http://127.0.0.1:8011") as client:
    assert client.post("/forecast", json={**payload, "p2_id": payload["p1_id"]}).status_code == 422
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(lambda _: client.post("/forecast", json=payload), range(8)))
    assert all(reply.status_code == 200 and reply.json() == image for reply in replies)
print("Offline, local HTTP, and container forecasts agree. Eight concurrent replies match.")
PY
docker stop "$container"
```

## Tell a live process from a ready one

Use a corrupt copy on port 8012 so the working process can keep serving.

```bash
mkdir -p "$work/broken"
cp "$work/first/selection.json" "$work/broken/"
printf 'corrupt model\n' > "$work/broken/model.joblib"
ACEBET_FORECAST_RESULTS="$work/matches.parquet" \
ACEBET_FORECAST_BUNDLE="$work/broken" \
uv run --locked --group mlops uvicorn acebet.app.historical:create_app \
  --factory --host 127.0.0.1 --port 8012
```

The server log should say that `model.joblib` differs from the frozen
selection. Use `curl -i` from another terminal to check `/healthz` and
`/readyz` on port 8012. Liveness returns 200 because the process can answer.
Readiness and valid forecasts return 503 because startup could not verify
the assets. Docker uses readiness for its health check. Stop the broken
process with Ctrl+C. The tests repeat the exercise with mismatched history
and a changed feature contract.

```bash
uv run --locked --group mlops --extra tracing pytest -q tests/test_historical_serving.py
```

## Measure a blocked event loop

Add two temporary handlers to the same one-worker app. Each sleeps
for one second. Put the lab in the scratch directory.

```bash
cat > "$work/forecast_lab.py" <<'PY'
from time import sleep

from acebet.app.historical import create_app

app = create_app()

@app.get("/block")
async def block():
    sleep(1)
    return {"done": True}

@app.get("/thread")
def thread():
    sleep(1)
    return {"done": True}
PY
ACEBET_FORECAST_RESULTS="$work/matches.parquet" \
ACEBET_FORECAST_BUNDLE="$work/first" \
uv run --locked --group mlops uvicorn forecast_lab:app --app-dir "$work" \
  --host 127.0.0.1 --port 8012 --workers 1
```

From another terminal, start the slow request and time a second request to
`/healthz` about 200 ms later.

```bash
uv run --locked --group mlops python - <<'PY'
import asyncio
from time import perf_counter

import httpx

async def measure():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8012") as client:
        (await client.get("/healthz")).raise_for_status()
        for route in ("/block", "/thread") * 3:
            slow = asyncio.create_task(client.get(route))
            await asyncio.sleep(0.2)
            start = perf_counter()
            (await client.get("/healthz")).raise_for_status()
            print(f"{route:8} overlapping /healthz: {1000 * (perf_counter() - start):.0f} ms")
            (await slow).raise_for_status()

asyncio.run(measure())
PY
```

Expect about 800 ms behind `/block` and a response close to idle latency
behind `/thread`. In the async handler, `sleep(1)` ties up the event loop
that dispatches requests. FastAPI runs a regular `def` handler in a thread
pool, so the loop keeps dispatching while a thread waits. An
`await asyncio.sleep(1)` would also yield the loop while waiting.
[FastAPI async handlers and threads](https://fastapi.tiangolo.com/async/)

The real forecast handler uses `def` because replay and prediction are
synchronous. The sleep exercise measures responsiveness during a wait. For
CPU work in CPython, the global interpreter lock (GIL) lets one thread at a
time execute Python bytecode in each standard interpreter. Some native
numerical code can release the GIL. Extra Uvicorn workers run as separate
processes and each loads a copy of the model and history. Measure forecast
latency, throughput, and memory before choosing a worker count.
[Python threading](https://docs.python.org/3.12/library/threading.html) · [Uvicorn deployment](https://www.uvicorn.org/deployment/)

Stop the lab and local processes with Ctrl+C. Keep the request, response,
asset checksums, and image ID. These identify the forecast to investigate
when a deployment gives an unexpected result in the
[monitoring and rollback drill](./diagnose-and-roll-back-a-forecast.md).
