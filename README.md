# Mastering Python Project Management with UV: MLOps

AceBet is a FastAPI-based MLOps sample project that demonstrates how to use UV for dependency management, testing, CI/CD, and Dockerized deployment.

## Quickstart

```bash
cp .env.example .env
# set ACEBET_SECRET_KEY to a strong generated value in .env
uv sync
export ACEBET_SECRET_KEY="replace-with-a-long-random-secret"
uv run pytest tests
uv run fastapi run src/acebet/app/main.py --host 0.0.0.0 --port 8000
```

Generate a strong key (pick one):

```bash
python -c "import secrets; print(secrets.token_urlsafe(64))"
# or
openssl rand -hex 64
```

Expected quickstart result:
- Environment file exists at `.env` and contains a strong `ACEBET_SECRET_KEY`.
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


## Runtime configuration precedence

AceBet resolves configuration in this order:

1. Process environment (CI/container/runtime)
2. Local `.env` for development
3. Code defaults only for non-sensitive values

At startup, the API emits a debug log with the effective non-secret configuration values to aid troubleshooting.
