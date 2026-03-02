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
git clone <your-repo-url> mlops-uv
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

### 4. Start the API

```bash
uv run fastapi run src/acebet/app/main.py --host 0.0.0.0 --port 8000
```

## Expected output

- `uv sync` completes with a success message and creates/updates `.venv`.
- `uv run pytest tests` shows all tests passing.
- `uv run fastapi run ...` shows a running server and a local URL such as `http://0.0.0.0:8000`.

## Troubleshooting

- **`No module named ...` during run/test**: re-run `uv sync` to ensure the environment is fully installed.
- **Port 8000 already in use**: change the port, for example `--port 8001`.
- **Tests fail unexpectedly**: run with verbose output using `uv run pytest tests -vv` to inspect stack traces.
