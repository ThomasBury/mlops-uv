FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency manifests first for better layer caching.
COPY pyproject.toml uv.lock ./

# Install project dependencies (without installing the project itself yet).
RUN uv sync --frozen --no-cache --no-install-project

# Copy application source.
COPY src ./src

# Install the local project into the existing virtual environment.
RUN uv sync --frozen --no-cache

# Use an unprivileged user at runtime.
RUN addgroup --system app && adduser --system --ingroup app app && chown -R app:app /app
USER app

# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "src/acebet/app/main.py", "--port", "80", "--host", "0.0.0.0"]
