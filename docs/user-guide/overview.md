# User Guide Overview

This page explains how AceBet is organized and how to work with the core project workflow.

## Prerequisites

- Completed [Getting Started](../getting-started.md)
- Local environment synced with `uv sync`

## Project structure

```text
src/acebet/
  app/main.py                     # FastAPI app and routes
  app/dependencies/*.py           # Auth, schemas, prediction helpers
  train/train.py                  # Model training script
  dataprep/dataprep.py            # Data preparation helpers
tests/
  test_acebet.py                  # API and behavior tests
```

## Common development workflow

### 1. Sync dependencies

```bash
uv sync
```

### 2. Run tests before changes

```bash
uv run pytest tests
```

### 3. Implement feature or fix

```bash
# edit code in src/acebet or tests
```

### 4. Re-run tests

```bash
uv run pytest tests
```

### 5. Run API locally for manual verification

```bash
uv run fastapi run src/acebet/app/main.py --host 0.0.0.0 --port 8000
```

## Expected output

- Tests pass before and after your change.
- FastAPI starts and serves HTTP responses on the configured port.
- The edited code path is covered by existing or new tests.

## Troubleshooting

- **Flaky tests**: run a focused test module, e.g. `uv run pytest tests/test_acebet.py -vv`.
- **Import errors after refactor**: ensure imports still point to `acebet.*` modules.
- **Runtime mismatch**: verify Python version with `uv run python --version`.
