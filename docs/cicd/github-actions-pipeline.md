# CI/CD

The repository currently has two GitHub Actions workflows: one for the Python
test suite and one for the Zensical documentation site.

## Test Workflow

`.github/workflows/main.yaml` runs on pushes to `main` and currently does this:

```yaml
- uses: actions/checkout@v4
- uses: astral-sh/setup-uv@v5
- run: uv sync --all-extras --dev
- run: uv run pytest tests
```

That is the current CI contract. It does not run Docker smoke tests, docs
builds, or extra health/readiness checks.

## Docs Workflow

`.github/workflows/docs.yaml` builds the Zensical site when docs-related files
change, and deploys the generated `site/` directory to GitHub Pages on pushes
to `main` or `master`.

The build job now matches the repo layout:

```yaml
- uses: astral-sh/setup-uv@v5
- run: uv sync --frozen
- run: uv run zensical build
```

## Recommended Local Checks

```bash
uv sync
just test
just docs-build
just build
```

This page prefers the maintained `just` entrypoints where they exist, so local
checks stay aligned with the repo workflow instead of duplicating lower-level
tool invocations in multiple places.

## When To Refresh The Lockfile

Run `uv lock` after changing Python dependencies, including docs tooling such as
Zensical, then commit the updated `uv.lock`.
