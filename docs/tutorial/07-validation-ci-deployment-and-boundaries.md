---
title: Tutorial 7 - Validation And Release Boundaries
description: Run the non-mutating review gate, understand Docker smoke coverage, and keep tutorial claims aligned with executable contracts.
---

# Tutorial 7: Validation And Release Boundaries

The final step is to prove that data checks, model selection, artifacts,
serving, packaging, notebook code, and documentation still agree.

## Why This Exists

Validation is the final operational check: it asks whether data, decisions,
artifacts, API behavior, and documentation still agree. The smallest command
below proves that agreement without changing tracked files.

## Run The Repository Gate

```bash
just review
```

`review` is non-mutating and independent of a Docker daemon. It runs:

- prepared-data validation, a temporary raw-sample rebuild, and temporary
  player-state generation;
- constant/Elo/LightGBM training and selection without export;
- all tests, including wheel/sdist content and isolated-install tests, with
  branch coverage;
- the 80% overall coverage floor and 100% critical-module floor;
- Ruff lint and format checks plus `ty`;
- `jupyter execute` without `--inplace`;
- the Zensical docs build;
- a temporary authenticated API smoke using the real packaged bundle.

The notebook runs with a non-interactive Matplotlib backend. A cell error fails
the command, while the committed notebook retains null execution counts and no
outputs.

## Docker Is A Separate Runtime Proof

```bash
just docker-smoke
```

Docker is optional for tutorial readers and is intentionally excluded from
`review`. `docker-build` only builds the image. `docker-smoke` builds it and
proves that only the non-editable virtual environment enters the runtime, the
process is non-root, health and readiness pass, token authentication works,
caller identity resolves, and production prediction works with both packaged
fallback and explicit read-only mounted assets.

This remains a single-service image. Multi-stage builds construct an image;
Compose coordinates multiple running containers. Training, documentation,
promotion, and local MLflow logging are commands, not services, so adding
Compose would not improve this tutorial. If a trainer is containerized later,
give it a separate build target or image. Add Compose only when external MLflow,
a database, collectors, dashboards, or similar long-running dependencies must
run together locally.

!!! info "Shortcut"

    **Why it is enough here:** one image proves the API can run with its
    packaged bundle. **Add it when…** local MLflow, a database, collectors, or
    dashboards must run together; then add Compose. See
    [Tutorial Boundaries](../tutorial-shortcuts-vs-production.md).

## What Contract Tests Prevent

Tutorial regressions compare the canonical documented response with a real
packaged-bundle response. They also bind documented metrics, filenames, hashes,
selected kind, and reason to the committed metadata and promotion pointer.
They reject obsolete recipes, unsafe canonical-data output examples, stale lock
flags, and inaccurate Docker claims.

CI repeats the Python checks across 3.10–3.12, combines unit and integration
coverage on 3.12, runs notebook execution, verifies tracked files remain
unchanged, and keeps Docker smoke in its own job. The docs workflow explicitly
runs `uv sync --locked --dev` before building. Pull-request docs builds have
read-only permissions; only the main-branch deployment job receives Pages and
OIDC write permissions.

This is an executable tutorial release boundary, not automated application
deployment. The single
[Tutorial Boundaries](../tutorial-shortcuts-vs-production.md) matrix states
what remains outside the repository, including centralized observability,
registry infrastructure, orchestration, scanning/signing, and automated
application deployment.

## Experiment: Override The Published Port

**Prerequisites:** install `just` and work from the repository root. Neither a
Docker daemon nor `.env` is needed for this dry run.
**Prediction:** the override changes the host port while publication remains
restricted to `127.0.0.1`.

```bash
ACEBET_DOCKER_PORT=18080 just --dry-run docker-run
```

Expected checkpoint: among the printed commands is:

```text
docker run --rm --env-file .env -p "127.0.0.1:18080:8000" "acebet"
```

**Explain:** why does the host bind address matter even though the container
listens on `0.0.0.0`? What can this dry run establish, and what still requires
`just docker-smoke` against a running daemon?
**Cleanup:** the override applies to this command only; no container starts and
no files change.
