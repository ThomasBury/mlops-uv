# 7. Serve the frozen forecast

The final score was computed offline. A reader needs one forecast over HTTP. Before introducing a server, replay two fixtures with the installed command. F01 has a hand-checkable rating gap. Fixture `424816` (Barrere–Pospisil in Metz) also has an independently recorded batch prediction in `first/final/forecasts.csv`.

## Replay two requests offline

Use the same shell and `$work`. The JSON sent by a caller contains each player’s ID, the surface and cutoff. Keep these request files separate from the feature trace and replies.

```bash
cat > "$work/F01-request.json" <<'JSON'
{"fixture_id":"F01","p1_id":"P01","p2_id":"P02","surface":"Clay","cutoff":"2030-06-10T12:00:00Z"}
JSON
cat > "$work/424816-request.json" <<'JSON'
{"fixture_id":"424816","p1_id":"1985","p2_id":"427","surface":"Hard","cutoff":"2021-09-19T23:59:59Z"}
JSON
for id in F01 424816; do
  .venv/bin/python -m acebet.historical_forecast "$work/matches.parquet" "$work/first" \
    < "$work/$id-request.json" > "$work/$id-offline.json"
  cat "$work/$id-offline.json"
done
```

Follow `FrozenForecast.predict` in `src/acebet/historical_forecast.py`. It verifies the fixture and cutoff, sorts the IDs, and recovers either F01’s synthetic history or the archive’s eligible results. The fitted pipeline takes exactly one row containing `elo_gap` and `surface`. Its classes are checked at load time. Column 1 of `predict_proba` is the chance of a win for the smaller ID. Reversing the request flips both the gap and the probability.

```python
probability = float(self.model.predict_proba(inputs)[0, 1])
if request.p1_id != players[0]:
    gap, probability = -gap, 1.0 - probability
```

For F01 the gap is −16, the Elo probability is 0.4770, and the fitted logistic prediction is 0.469100. Find `424816` in the final CSV and compare its `logistic_tuned` value with the offline forecast (about 0.527281).

## Follow the same request through HTTP

Start a local worker from the checkout. The lifespan hook calls `load_forecaster` once at startup. It checks history, feature code, dependency versions, model bytes and the F01 reference prediction before storing the loaded forecaster. Load only trusted Joblib bundles.

```bash
export ACEBET_FORECAST_RESULTS="$work/matches.parquet"
export ACEBET_FORECAST_BUNDLE="$work/first"
.venv/bin/uvicorn acebet.app.historical:create_app --factory --host 127.0.0.1 --port 8010 \
  > "$work/server.log" 2>&1 &
server_pid=$!
base=http://127.0.0.1:8010
curl -fsS --retry 12 --retry-connrefused "$base/readyz"
for id in F01 424816; do
  curl -fsS -H 'Content-Type: application/json' --data-binary @"$work/$id-request.json" \
    "$base/forecast" > "$work/$id-http.json"
done
```

FastAPI parses JSON into `ForecastRequest`. The synchronous `def forecast(...)` handler gets the already loaded model, calls `forecaster.predict(payload, ...)` and returns a `ForecastResponse`. An event loop can handle other sockets while one request waits for I/O. A blocking call on that loop would delay them. FastAPI dispatches this handler to a thread pool, where lookup and inference run synchronously. Replaying many ratings is CPU work. On this CPython runtime the global interpreter lock (GIL) allows one thread per interpreter to execute Python bytecode at a time. Some native numerical routines can release it. Check the [optional timing lab](../labs/event-loop.md) to see what blocking the loop does to another request. [FastAPI execution model](https://fastapi.tiangolo.com/async/)

Check every identity and the probability for both fixtures. Save this as `compare-replies.py` in your scratch directory for the image exercise. If it fails, the assertion names the fixture and field to inspect.

```python
import json
from math import isclose
from pathlib import Path
import sys

work = Path(__file__).parent
for fixture in ("F01", "424816"):
    expected, observed = [
        json.loads((work / f"{fixture}-{mode}.json").read_text())
        for mode in sys.argv[1:]
    ]
    assert expected.keys() == observed.keys(), fixture
    for key, value in expected.items():
        if key == "p1_win_probability":
            assert isclose(value, observed[key], rel_tol=0, abs_tol=1e-8), (fixture, key)
        else:
            assert value == observed[key], (fixture, key)
    print(fixture, observed["p1_win_probability"], observed["model_sha256"])
```

Run the comparison from the checkout, then try a request with missing fields.

```bash
.venv/bin/python "$work/compare-replies.py" offline http
curl -sS -o "$work/invalid.json" -w '%{http_code}\n' -H 'Content-Type: application/json' \
  -d '{}' "$base/forecast"
kill "$server_pid"
```

The empty request returns 422. Now damage a **copy** of the model. Predict which endpoint will still work.

```bash
mkdir "$work/broken"
cp "$work/first/selection.json" "$work/broken/"
printf 'corrupt model\n' > "$work/broken/model.joblib"
ACEBET_FORECAST_BUNDLE="$work/broken" .venv/bin/uvicorn acebet.app.historical:create_app \
  --factory --host 127.0.0.1 --port 8012 > "$work/broken.log" 2>&1 &
broken_pid=$!
curl -fsS --retry 12 --retry-connrefused http://127.0.0.1:8012/healthz
curl -sS -o "$work/unready.json" -w '%{http_code}\n' http://127.0.0.1:8012/readyz
kill "$broken_pid"
```

Liveness returns 200 because the process can answer. Readiness returns 503 because the service detected the changed model. Look for `forecast_assets_invalid` in the log. Keep both successful HTTP replies for the [runtime comparison](./reproduce-the-forecast-runtime.md).
