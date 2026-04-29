# AceBet Documentation

This site documents the current `acebet` codebase as it runs today. It is a
small classical MLOps tutorial: validate a prepared ATP dataset, derive player
state, train a LightGBM pipeline, and serve predictions through FastAPI.

## Quickstart

```bash
uv sync
just data-check
just feature-state
just train
just test
just serve
```

Open `http://127.0.0.1:8000/docs` for the generated OpenAPI UI once the local
server is running.

## What To Read

- [Getting Started](./getting-started.md) for a first local run.
- [Classical MLOps Tutorial](./classical-mlops-tutorial.md) for the end-to-end workflow.
- [Developer Workflow](./uv/uv-workflows-for-mlops.md) for how `uv` and `just` divide responsibilities in this repo.
- [User Guide Overview](./user-guide/overview.md) for repository structure and commands.
- [Training and Inference](./user-guide/training-and-inference.md) for artifact and feature-contract details.
- [API Endpoints](./api/sync-endpoints.md) for the current HTTP surface.
- [CI/CD](./cicd/github-actions-pipeline.md) and [Deployment](./deployment/docker-and-github-actions.md) for automation and container behavior.
- [Tutorial Boundaries](./tutorial-shortcuts-vs-production.md) for the shortcuts this repo takes on purpose.

## Docs Authoring

The authored docs source lives in `docs/` and is built by Zensical using
`zensical.toml`.

```bash
just docs-serve
just docs-build
```
