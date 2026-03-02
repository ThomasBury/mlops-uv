# Mastering Python Project Management with UV: MLOps

AceBet is a FastAPI-based MLOps sample project that demonstrates how to use UV for dependency management, testing, CI/CD, and Dockerized deployment.

## Quickstart

```bash
uv sync
uv run pytest tests
uv run fastapi run src/acebet/app/main.py --host 0.0.0.0 --port 8000
```

Expected quickstart result:
- Dependencies install successfully.
- Test suite passes.
- API starts locally and responds at `http://localhost:8000`.

## Full documentation

- [Docs Index](docs/index.md)
- [Getting Started](docs/getting-started.md)
- [User Guide Overview](docs/user-guide/overview.md)
- [Training and Inference](docs/user-guide/training-and-inference.md)
- [API: Sync Endpoints](docs/api/sync-endpoints.md)
- [API: Async Behavior](docs/api/async-behavior.md)
- [UV Workflows for MLOps](docs/uv/uv-workflows-for-mlops.md)
- [GitHub Actions Pipeline](docs/cicd/github-actions-pipeline.md)
- [Deployment: Docker and GitHub Actions](docs/deployment/docker-and-github-actions.md)
