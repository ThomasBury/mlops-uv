# UV Workflows for MLOps

Use UV to standardize environment creation, dependency locking, testing, and developer tooling.

## Prerequisites

- UV installed (`uv --version`)
- Python 3.12+
- Access to this repository and `pyproject.toml`

## Core workflows

### 1. Initialize/sync project environment

```bash
uv sync
```

### 2. Add runtime dependencies

```bash
uv add fastapi scikit-learn pandas lightgbm
```

### 3. Add development dependencies

```bash
uv add --dev pytest
```

### 4. Refresh lockfile after dependency changes

```bash
uv lock
```

### 5. Run tests in the managed environment

```bash
uv run pytest tests
```

### 6. Run one-off tools without persistent installation

```bash
uvx pytest --version
```

## Expected output

- `uv sync` installs pinned dependencies from lock data.
- `uv add` updates `pyproject.toml` and lock metadata.
- `uv run pytest tests` executes tests using the project virtual environment.

## Troubleshooting

- **Lock mismatch warnings**: run `uv lock` then `uv sync`.
- **Unexpected package version**: inspect constraints in `pyproject.toml`.
- **Command resolution issues**: use `uv run <cmd>` to ensure tools are run in `.venv` context.
