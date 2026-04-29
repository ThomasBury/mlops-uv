# Developer Workflow

This repository uses `uv` and `just` together on purpose.

- Use `uv` for Python environment management, dependency changes, and lockfile updates.
- Use `just` for repeatable repository workflows such as testing, docs builds, training, and serving.

That split keeps the low-level packaging concerns in one tool and the project
command surface in another.

## Why Both Tools Exist

`uv` is the source of truth for:

- installing the locked environment
- editing dependencies in `pyproject.toml`
- refreshing `uv.lock`
- running Python commands inside the managed environment

`just` is the source of truth for:

- the standard contributor workflows
- multi-step tasks that should stay stable across docs and local usage
- named entrypoints such as `test`, `serve`, `train`, and `docs-build`

The docs should prefer `just` whenever the repository already exposes a recipe
for the task being described.

## Common Commands

```bash
uv sync
just test
just docs-build
just serve
just data-check
just feature-state
just train
just build
```

Those are the commands contributors will use most often in this tutorial.

## When To Use `uv` Directly

Call `uv` directly when you are managing the Python toolchain rather than
running a repo workflow.

Typical examples:

```bash
uv sync
uv lock
uv add zensical --dev
uv run python -m acebet.train.train --help
```

Use direct `uv run ...` commands in docs only when one of these is true:

- there is no maintained `just` recipe yet
- the page is documenting an internal or one-off command
- the exact underlying Python invocation is itself the thing being explained

## When To Prefer `just`

Prefer `just` when the repo already has a recipe that names the workflow.

Examples:

- `just test` instead of `uv run pytest tests`
- `just docs-build` instead of `uv run zensical build`
- `just build` instead of `uv build`
- `just serve` instead of repeating the local FastAPI dev command

That reduces documentation drift and keeps CI notes, README examples, and local
usage aligned with the same maintained entrypoints.

## When To Add A New Recipe

Add a `just` recipe when:

- the same command sequence appears in multiple docs pages
- a workflow has multiple steps that should be run consistently
- contributors are expected to repeat the task regularly

Do not add a recipe just to wrap an obscure one-off command that is only useful
for debugging a single internal detail.

## Keeping Docs And Recipes Aligned

When a new repo workflow is important enough to document repeatedly:

1. Add or update the `just` recipe.
2. Prefer that recipe in README and docs examples.
3. Keep `uv` references focused on dependency and lockfile management.

That is the intended documentation style for this tutorial.
