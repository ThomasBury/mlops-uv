# Deployment: Docker and GitHub Actions

This guide combines containerization and GitHub Actions for repeatable deployments.

## Prerequisites

- Docker installed locally
- Existing `Dockerfile` in repository root
- GitHub repository with Actions enabled
- Container registry credentials configured as GitHub secrets
- Runtime configuration stored in a local `.env` file (not in the image)

## Runtime environment variables

Pass secrets and environment-specific config at runtime with `--env-file .env`.
Do **not** bake secrets into Docker images with `ARG` or hardcoded `ENV` values.

Example `.env` file for local testing:

```dotenv
ACEBET_SECRET_KEY=replace-with-a-long-random-value
ACEBET_ENV=production
ACEBET_DEBUG=false
```

## Build and run locally with Docker

### 1. Build image

```bash
docker build -t acebet:local .
```

### 2. Run container with runtime env

```bash
docker run --rm --env-file .env -p 8000:80 acebet:local
```

### 3. Validate service response

```bash
curl http://localhost:8000/
```

### Optional: Docker Compose runtime env example

```yaml
services:
  acebet:
    image: acebet:local
    env_file:
      - .env
    ports:
      - "8000:80"
```

## GitHub Actions deployment skeleton

```yaml
name: Build and Push

on:
  push:
    branches: ["main"]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository_owner }}/acebet:latest
```

## Mapping GitHub secrets to runtime variables

Use repository or environment secrets for runtime variables, and inject them when starting the container.

| Runtime variable | GitHub secret (example) | Notes |
| --- | --- | --- |
| `ACEBET_SECRET_KEY` | `ACEBET_SECRET_KEY` | Required. Keep unique per environment and rotate regularly. |
| `ACEBET_ENV` | `ACEBET_ENV` | Example values: `staging`, `production`. |
| `ACEBET_DEBUG` | `ACEBET_DEBUG` | Usually `false` outside local development. |

Example deployment step that writes a temporary env file on the runner and passes it at runtime:

```yaml
- name: Run container with runtime secrets
  run: |
    cat > runtime.env <<'ENVVARS'
    ACEBET_SECRET_KEY=${{ secrets.ACEBET_SECRET_KEY }}
    ACEBET_ENV=${{ secrets.ACEBET_ENV }}
    ACEBET_DEBUG=${{ secrets.ACEBET_DEBUG }}
    ENVVARS

    docker run -d --name acebet --env-file runtime.env -p 8000:80 \
      ghcr.io/${{ github.repository_owner }}/acebet:latest
```

## Security hygiene for tutorial workflows

- **Do not commit `.env` files**: add `.env` to `.gitignore`, and keep only a non-sensitive `.env.example` in version control.
- **Rotate leaked keys immediately**: if `ACEBET_SECRET_KEY` (or any credential) is exposed in commit history, logs, screenshots, or chat, generate a new key and update all environments/secrets.
- **Prefer least privilege**: only expose the variables the container needs at runtime.

## Expected output

- Local Docker image builds without dependency/install errors.
- Container starts and serves API responses on mapped port.
- GitHub workflow builds and pushes tagged images to the configured registry.
- Runtime variables (especially `ACEBET_SECRET_KEY`) are injected at container start, not image build.

## Troubleshooting

- **Docker build fails at `uv sync --frozen`**: ensure `uv.lock` is up-to-date and committed.
- **Container starts but endpoint unavailable**: verify container port and host mapping (`-p 8000:80`) and that required vars are present in `.env`.
- **App fails due to missing secret key**: confirm `ACEBET_SECRET_KEY` exists in local `.env` and GitHub secrets.
- **Registry push denied**: validate login credentials and repository write permissions.
