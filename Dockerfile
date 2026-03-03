# syntax=docker/dockerfile:1.4

FROM python:3.12.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Pin uv to a specific version for reproducible builds.
COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /uvx /bin/

WORKDIR /app

# Copy only what uv needs to resolve dependencies.
# README.md and LICENSE are intentionally excluded here to avoid
# invalidating the dependency cache on every doc edit.
COPY pyproject.toml uv.lock ./

# Install dependencies without installing the local project yet.
# This layer is cached as long as pyproject.toml and uv.lock don't change.
RUN uv sync --frozen --no-cache --no-install-project

# Copy source and data only after dependencies are cached.
COPY src ./src
COPY data ./data

# Install the local project into the existing virtual environment.
RUN uv sync --frozen --no-cache


FROM python:3.12.10-slim AS runtime

# Activate the venv globally so python/fastapi are on PATH without hardcoding.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Create unprivileged user before copying files so we can chown in one pass.
RUN addgroup --system app && adduser --system --ingroup app app

# Copy only runtime artifacts from builder, owned by app from the start.
# This avoids a separate chown -R pass (which creates an extra layer).
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/src  /app/src
COPY --from=builder --chown=app:app /app/data /app/data

USER app

EXPOSE 80

CMD ["fastapi", "run", "src/acebet/app/main.py", "--port", "80", "--host", "0.0.0.0"]
