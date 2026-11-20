---
title: CI/CD
description: Reference the Python matrix, integration coverage, notebook execution, Docker smoke, and docs workflow.
---

# CI/CD

AceBet has a main validation workflow and one documentation workflow.

## Main Workflow

Python 3.10, 3.11, and 3.12 each run:

```bash
uv sync --locked --dev
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check
uv run pytest tests -m "not (api or tracing or packaging)"
```

Python 3.12 uploads unit branch coverage. A dependent 3.12 integration job
runs the temporary data/feature checks, non-exporting selection, and
`just notebook-check`; installs tracing with
`uv sync --locked --dev --extra tracing`; then runs API, tracing, packaging,
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

It builds Zensical for relevant pull requests with only read access. Pushes to
`main` additionally upload the Pages artifact, then a dependent deployment job
uses the `github-pages` environment and alone receives `pages: write` and
`id-token: write`. Pages concurrency prevents overlapping deployments.

In the repository settings, open **Pages** and set **Build and deployment →
Source** to **GitHub Actions**. The deployment job checks the public homepage
for the Zensical generator and the **Start Here** heading. It also checks that
the tutorial and setup routes serve their expected pages. The job fails if
these checks still fail after six attempts.

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
