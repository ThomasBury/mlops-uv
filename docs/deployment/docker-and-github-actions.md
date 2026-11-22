# Container targets

| Target | Assets and entry point | Runtime check |
| --- | --- | --- |
| `historical-forecast` | Named `forecast_assets` build context supplies `matches.parquet`, `selection.json` and `model.joblib`. Uvicorn serves `acebet.app.historical:create_app`. | Health check uses `/readyz`. Compare F01 and 424816 with offline and local HTTP replies. |
| `release` (default) | Installed wheel supplies packaged state and the promoted Elo bundle. FastAPI serves `acebet.app.main:app`. | Health check uses `/healthz`. Smoke tests check readiness and authenticated prediction. |

For Docker Engine/Buildx and native Podman commands using your own historical assets, follow the [runtime lesson](../tutorial/reproduce-the-forecast-runtime.md). The remaining commands operate on the packaged predictor.

## Packaged smoke test

```bash
just docker-build
just docker-smoke
```

`docker-build` only runs `docker build -t acebet .`.
`docker-smoke` builds the same image and then verifies a non-root process,
`/healthz`, `/readyz`, token authentication, `/users/me`, and an
authenticated production prediction.

## Exact Builder Stages

Both stages use the same digest-pinned
`python:3.12.13-slim-bookworm` image. The builder copies only the `uv` and
`uvx` binaries from `ghcr.io/astral-sh/uv:0.9.30`; it does not inherit a
second Python runtime. It performs two locked, cached, non-editable syncs:

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project --no-editable

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --reinstall-package acebet
```

The first operation installs locked third-party dependencies without the
project. The second installs the copied project into `/app/.venv` while still
excluding dev dependencies. `--reinstall-package acebet` rebuilds the local
wheel when application files change, even with a warm uv cache. Because the
install is non-editable, the runtime
does not need the source tree.

The runtime installs `libgomp1` and copies only `/app/.venv` from the builder;
application code and immutable packaged assets are already installed inside
that environment. It runs one FastAPI process as non-root `appuser` through the
configured entrypoint:

```bash
fastapi run --entrypoint acebet.app.main:app --host 0.0.0.0 --port 8000
```

Its standard-library healthcheck calls `/healthz`; curl is not installed.

## Configuration And Assets

`.env` is excluded from the build context. Supply configuration at runtime:

```bash
just docker-run
ACEBET_DOCKER_PORT=18080 just docker-run
```

Both run recipes publish only on `127.0.0.1`, including overridden host ports.
The smoke recipe also publishes on loopback (`ACEBET_DOCKER_SMOKE_PORT`, default
`18082`, and the next port for mounted assets). The process inside the container
continues to listen on `0.0.0.0:8000`.

Root `data/`, `models/`, tests, site output, and MLflow output are also
excluded. The wheel still contains the promoted schema-v2 Elo bundle and
packaged production and sample player state, so the image is ready without host
asset mounts. Those installed files are an immutable fallback tied to the
image.

With no overrides, production uses the packaged `promoted_model.json` bundle
and packaged `latest_player_stats.feather`. Set `ACEBET_MODEL_DIR` and/or
`ACEBET_DATA_DIR` to opt into external production assets. A configured model
directory must contain `promoted_model.json`; a configured data directory must
contain `latest_player_stats.feather`. Missing configured files fail startup
and never fall back. Testing mode always uses packaged sample player state and
the packaged promoted model, so production mounts cannot leak into the sample
route.

The local promotion-to-serving path is explicit:

```bash
just rebuild-artifacts models
just serve-with-assets
```

`just docker-run-with-assets` mounts root `data/` and `models/` read-only and
sets both directory variables to those mounts. `just docker-smoke` exercises
both the immutable packaged fallback and an explicit read-only mounted override.
