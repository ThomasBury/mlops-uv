---
title: CI/CD
description: Reference the Python matrix, integration coverage, notebook execution, Docker smoke, and docs workflow.
---

# CI/CD

## Historical forecast workflow

`.github/workflows/forecast.yaml` runs `just forecast-release-test` and `just forecast-candidate` under Python 3.12. Scheduled and manual runs also run `just forecast-test`. Pull requests, pushes to main, manual dispatch and Mondays at 06:23 UTC trigger a run. The job checks for a clean checkout and uploads the Parquet history, frozen bundle and validation record as `historical-forecast`. A valid comparison can complete this job successfully even when the gate rejects the candidate. Approval and starting a service are separate actions. [Release decision exercise](../tutorial/automate-and-audit-a-forecast.md)

## Main Workflow

Python 3.10, 3.11, and 3.12 each run:

```bash
uv sync --locked --dev --group mlops
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check
uv run pytest tests -m "not (api or tracing or packaging)"
```

Python 3.12 uploads unit branch coverage. A dependent 3.12 integration job
runs the temporary data/feature checks, non-exporting selection, and
`just notebook-check`; installs the forecast and tracing dependencies with
`uv sync --locked --dev --group mlops --extra tracing`; then runs API, tracing, packaging,
wheel, sdist, and isolated-install tests.

Unit and integration coverage are combined. CI enforces at least 80% overall
branch coverage and 100% for the critical artifact and prediction modules.
`git diff --exit-code` proves the checks did not change tracked files.

The Docker job runs `just docker-smoke`, not merely `docker-build`. It checks
the non-root runtime, health, readiness, authentication, caller identity, and a
production prediction.

## Documentation Workflow

`.github/workflows/docs.yaml` separates its build and deployment jobs. The
build installation step is exactly:

```bash
uv sync --locked --dev
```

Relevant pull requests build a clean Zensical site with read access and check
the lesson bookmark, static redirects, privacy link, search index and downloadable notebook.
On relevant pushes to `main`, the build uploads that checked `site/` directory.
The deployment job downloads it, confirms the target Cloudflare subdomain and
publishes the same files with Wrangler. Only this job uses the
`cloudflare-production` environment and its Cloudflare credentials.
It fetches the public lesson and sitemap and compares them with the artifact.
GitHub Actions is the production publisher for the tutorial.

Every third-party action is pinned to a full commit SHA with its release in a
comment. Dependabot checks GitHub Actions and Docker references weekly.

## Dependency And Lock Discipline

The pipeline does not use all extras. Notebook dependencies remain in
`dependency-groups.dev`; MLflow remains in `mlops`; OpenTelemetry is installed
only with the `tracing` extra.
Distribution tests prove those tools do not leak into the default wheel.

Run `uv lock` after an intentional dependency declaration change. Every CI
sync uses `--locked`, so a stale lock fails rather than being rewritten.

The closest local gate is `just review`; run `just docker-smoke` separately
when a Docker daemon is available.
