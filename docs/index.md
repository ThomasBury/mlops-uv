# Documentation Index

Welcome to the AceBet MLOps documentation hub. Use this page to start quickly, then dive into role-specific workflows.

## Start here

1. Read [Getting Started](./getting-started.md) to install tooling and run the project locally.
2. Read the [User Guide Overview](./user-guide/overview.md) to understand project architecture and workflow.
3. Follow [Training and Inference](./user-guide/training-and-inference.md) to execute model workflows.
4. Review API behavior in [Sync Endpoints](./api/sync-endpoints.md) and [Async Behavior](./api/async-behavior.md).
5. Use [UV Workflows for MLOps](./uv/uv-workflows-for-mlops.md) for dependency and environment management.
6. Set up automation via [GitHub Actions Pipeline](./cicd/github-actions-pipeline.md) and deployment via [Docker and GitHub Actions](./deployment/docker-and-github-actions.md).

## Role-based entry points

### ML engineer

- [Getting Started](./getting-started.md)
- [Training and Inference](./user-guide/training-and-inference.md)
- [UV Workflows for MLOps](./uv/uv-workflows-for-mlops.md)

### Backend engineer

- [Getting Started](./getting-started.md)
- [Sync Endpoints](./api/sync-endpoints.md)
- [Async Behavior](./api/async-behavior.md)
- [Docker and GitHub Actions](./deployment/docker-and-github-actions.md)

### Maintainer

- [User Guide Overview](./user-guide/overview.md)
- [GitHub Actions Pipeline](./cicd/github-actions-pipeline.md)
- [Docker and GitHub Actions](./deployment/docker-and-github-actions.md)
- [UV Workflows for MLOps](./uv/uv-workflows-for-mlops.md)

## Common prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Git
- Docker (for container workflows)
- GitHub repository access (for CI/CD)

## Quick verification commands

```bash
uv --version
uv sync
uv run pytest tests
```

### Expected output

- `uv --version` prints an installed version (for example, `uv 0.x.x`).
- `uv sync` completes without dependency resolution errors.
- `uv run pytest tests` reports passing tests.

## Troubleshooting

- **`uv: command not found`**: install UV and restart your shell.
- **Dependency resolution conflicts**: run `uv lock --upgrade` and re-run `uv sync`.
- **Docker build fails on dependency install**: ensure `uv.lock` is committed and use `uv sync --frozen` in container builds.
