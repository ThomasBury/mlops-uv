---
title: Getting Started
description: Install the locked environment, run the API, and use the non-mutating validation commands.
---

# Getting Started

Use this guide to try the packaged model or run maintainer checks. To train
and serve your own bundle, start with the [ten-lesson tutorial](./tutorial/index.md).

Install Python `>=3.10,<3.13`, `uv`, `just`, Git, and `curl`. Docker is optional
unless you want to run the container smoke.

## Clean-Clone Setup

```bash
uv sync --locked
[ -f .env ] || cp .env.example .env
```

`--locked` requires the committed `uv.lock` to match
`pyproject.toml`. Do not refresh the lock merely because setup failed:
investigate the reported Python, platform, cache, network, or lock mismatch.
Run `uv lock` only when dependency declarations intentionally change, then
review and commit the resulting lock diff.

Dependencies are locked so the checks use the reviewed
environment. See [Developer Workflow](./uv/uv-workflows-for-mlops.md) when you
need the `uv` and `just` details.

## Start The API

```bash
just serve
```

The development server listens on `http://127.0.0.1:8000`; OpenAPI is at
`/docs`. The copied development configuration explicitly opts into the
tutorial-only insecure signing key. Production must supply
`ACEBET_SECRET_KEY` and an explicit `ACEBET_DEMO_PASSWORD` of at least 12
characters. Passwords in every environment must be at most 72 UTF-8 bytes.

Leave the server running. In a second terminal at the repository root, request
an authenticated prediction from the packaged model.

```bash
just predict-production
```

## Highest-Value Validation

Run the test suite on its own with `just test`, or run the full local gate.

```bash
just review
```

This non-mutating gate covers data checks, non-exporting candidate selection,
tests and packaging/install checks with coverage, Ruff, `ty`, notebook
execution, docs, and a temporary authenticated API smoke.

```bash
just docker-smoke
```

Docker smoke is separate because Docker is optional and `review` does not
require a daemon.

!!! info "Shortcut"

    The local API is the only long-running service. Compose becomes useful
    when MLflow, a database, or telemetry backends must run together locally.

## Safe Raw And Notebook Exercises

```bash
just data-check-sample
just data-build data/raw/atp_matches_sample.csv var/tutorial/atp_data_sample.feather
just notebook-check
```

The raw sample has only 100 rows and demonstrates transformations; it cannot
rebuild the 44,617-row production dataset. The generic builder always requires
an explicit input and output. Use ignored `var/` or temporary destinations,
never the canonical production Feather path.

## Intentional Artifact Regeneration

Packaged fallback assets let `just serve` start without root `data/` or
`models/` directories. If you intentionally need new local artifacts, use the
companion workflow:

```bash
just rebuild-artifacts
```

!!! warning "Mutating workflow"

    This refreshes player-state artifacts, exports a bundle, and promotes it
    under ignored local storage. It is not part of the sequential tutorial or
    `just review`.

## Documentation

`just docs-serve` previews the site on `127.0.0.1:8002`.
`just docs-build` writes the static site to `site/`.

If port 8000 is busy, use
`uv run fastapi dev --host 127.0.0.1 --port 8001`. For exact routes and
errors, see [API Endpoints](./api/sync-endpoints.md).
