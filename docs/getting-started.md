# Getting Started

This guide helps you get the AceBet MLOps project running locally with UV.

## Prerequisites

- Python 3.12+
- UV installed and available on `PATH`
- Git
- Optional: Docker for image builds

## Setup steps

### 1. Clone and enter the repository

```bash
git clone https://github.com/OWNER/REPOSITORY.git mlops-uv
cd mlops-uv
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Run tests

```bash
uv run pytest tests
```

### 4. Configure environment variables

Configuration is loaded with the following precedence:

1. Process environment (CI/container/runtime)
2. Local `.env` file for development
3. In-code defaults for non-sensitive values only

Set a required secret before starting the API:

```bash
export ACEBET_SECRET_KEY="replace-with-a-long-random-secret"
```

You can also place the same value in a local `.env` file for development.

### 5. Start the API

```bash
uv run fastapi run src/acebet/app/main.py --host 0.0.0.0 --port 8000
```

## Expected output

- `uv sync` completes with a success message and creates/updates `.venv`.
- `uv run pytest tests` shows all tests passing.
- `uv run fastapi run ...` shows a running server and a local URL such as `http://0.0.0.0:8000`.
- Startup logs include an `Effective config (secrets redacted)` debug message for troubleshooting.

## Troubleshooting

- **`No module named ...` during run/test**: re-run `uv sync` to ensure the environment is fully installed.
- **Port 8000 already in use**: change the port, for example `--port 8001`.
- **Tests fail unexpectedly**: run with verbose output using `uv run pytest tests -vv` to inspect stack traces.
