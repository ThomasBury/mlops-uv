# GitHub Actions Pipeline

This page describes a UV-based CI pipeline for testing AceBet on pushes and pull requests.

## Prerequisites

- GitHub repository with Actions enabled
- Workflow file under `.github/workflows/`
- UV-compatible `pyproject.toml` and lockfile committed

## Example workflow

Create `.github/workflows/testing.yml`:

```yaml
name: Testing

on:
  push:
    branches: ["main"]
  pull_request:

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install UV
        uses: astral-sh/setup-uv@v5
      - name: Sync dependencies
        run: uv sync --all-extras --dev
      - name: Run tests
        run: uv run pytest tests
```

## Validate pipeline locally before push

```bash
uv sync --dev
uv run pytest tests
```

## Expected output

- GitHub Actions job installs UV and project dependencies successfully.
- Test step completes with passing tests.
- Pull requests show green CI status when checks pass.

## Troubleshooting

- **Action cannot find UV command**: confirm `astral-sh/setup-uv` step runs before any `uv` commands.
- **CI dependency failures**: commit updated `uv.lock` and re-run workflow.
- **Tests pass locally but fail in CI**: compare Python/OS versions and pin needed dependencies.
