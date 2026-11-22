# Measure a blocked event loop

After lesson 7, use the same checkout and scratch directory. Free port 8012 first. This lab compares two handlers in a single worker while a second request arrives.



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
.venv/bin/uvicorn forecast_lab:app --app-dir "$work" \
  --host 127.0.0.1 --port 8012 --workers 1
```

From another terminal, start the slow request and time a second request to
`/healthz` about 200 ms later.

```bash
.venv/bin/python - <<'PY'
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


Stop the lab with Ctrl+C. [Return to serving](../tutorial/serve-the-frozen-forecast.md).
