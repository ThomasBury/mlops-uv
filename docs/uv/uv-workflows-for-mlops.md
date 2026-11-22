---
title: Developer Workflow
description: Understand the locked uv environment, dependency scopes, and stable just contributor commands.
---

# Developer Workflow

`uv` owns dependency declarations, resolution, the virtual environment, and
package execution. `just` owns the contributor-facing workflows.

## Locked Setup

```bash
uv sync --locked
```

The default `dev` group is synced for contributors. CI spells that intent
explicitly as `uv sync --locked --dev`. A lock mismatch fails instead of
rewriting `uv.lock`.

The local package cache also watches `src/acebet/__init__.py`, so a version
bump refreshes installed package metadata during sync.

Run `uv lock` only after an intentional dependency declaration change.
Do not use it as generic troubleshooting for a clean-clone setup failure.

## Dependency Scopes

- core dependencies install in the runtime wheel, including bare FastAPI, its
  CLI, Uvicorn, multipart form support, and HTTPX for installed-package smoke checks;
- `dev` contains coverage, pytest, Ruff, `ty`, Zensical,
  Matplotlib, `nbclient`, and `ipykernel`;
- `mlops` adds MLflow;
- FastAPI requires the OpenTelemetry API package; its native tracing, metrics,
  logs, operation spans, and automatic exporter configuration are disabled;
- the `tracing` extra adds the optional SDK, exporters, and instrumentation for
  AceBet's application-owned tracing integration.

```bash
uv sync --locked --group mlops
uv sync --locked --extra tracing
```

`uv sync --locked --no-dev --no-editable` excludes notebook and contributor
tooling and installs the project into the portable virtual environment copied
by Docker.

## Historical forecast commands

`just forecast-test` runs the cutoff, fitting, serving, monitoring and recovery checks. `just forecast-candidate RUN` creates a new reviewed historical bundle. `just forecast-promote RUN` applies the explicit Elo gate. To run one step, look up `forecast-prepare`, `forecast-train`, `forecast-evaluate` or `forecast-serve` in `just --list`. [Manual commands and artifacts](../reference/historical-forecast.md)

## Packaged predictor and repository checks

```bash
just data-check
just data-check-sample
just feature-state-check
just train-check
just notebook-check
just review
just docker-smoke
```

The lessons introduce the underlying commands before automating them. In contributor code, document public arguments, return values and failure cases when they affect callers.

`just review` is non-mutating and Docker-independent. Commands named
`refresh-*`, `train-local`, `promote-local`, or `rebuild-artifacts`
are explicit companion workflows that may create or replace local artifacts.
