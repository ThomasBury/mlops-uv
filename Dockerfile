FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
WORKDIR /app
COPY . /app

# Install runtime dependencies only.
RUN uv sync --frozen --no-cache --no-dev

EXPOSE 80

# Run the application.
CMD ["/app/.venv/bin/fastapi", "run", "src/acebet/app/main.py", "--port", "80", "--host", "0.0.0.0"]
