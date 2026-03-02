# Deployment: Docker and GitHub Actions

This guide combines containerization and GitHub Actions for repeatable deployments.

## Prerequisites

- Docker installed locally
- Existing `Dockerfile` in repository root
- GitHub repository with Actions enabled
- Container registry credentials configured as GitHub secrets

## Build and run locally with Docker

### 1. Build image

```bash
docker build -t acebet:local .
```

### 2. Run container

```bash
docker run --rm -p 8000:80 acebet:local
```

### 3. Validate service response

```bash
curl http://localhost:8000/
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

## Expected output

- Local Docker image builds without dependency/install errors.
- Container starts and serves API responses on mapped port.
- GitHub workflow builds and pushes tagged images to the configured registry.

## Troubleshooting

- **Docker build fails at `uv sync --frozen`**: ensure `uv.lock` is up-to-date and committed.
- **Container starts but endpoint unavailable**: verify container port and host mapping (`-p 8000:80`).
- **Registry push denied**: validate login credentials and repository write permissions.
