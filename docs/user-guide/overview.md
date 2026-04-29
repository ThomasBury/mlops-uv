# User Guide Overview

This page maps the repository to the commands you use most often.

## Project Structure

```text
src/acebet/
  app/main.py                     # FastAPI app, lifespan, and routes
  app/dependencies/auth.py        # JWT helpers and tutorial user store
  app/dependencies/data_models.py # Pydantic request/response schemas
  app/dependencies/predict_winner.py
  dataprep/dataprep.py            # Prepared-data validation and player-state generation
  features.py                     # Shared feature contract
  train/train.py                  # Training, evaluation, export, MLflow
tests/test_acebet.py              # API, data, feature, and training regression tests
docs/                             # Canonical authored documentation source
zensical.toml                     # Zensical site configuration
```

## Normal Workflow

```bash
uv sync
just data-check
just feature-state
just train
just test
just serve
```

Use `just manual` when you want the repo to start a temporary API and exercise
the authenticated HTTP routes end to end.

## Docs Workflow

```bash
just docs-serve
just docs-build
```

Those commands serve or build the Zensical site from the `docs/` tree.

## Runtime Notes

- The API ships with a demo user store in code: `johndoe` / `secret`.
- The JWT signing key is also defined in code today.
- Production predictions prefer `data/latest_player_stats.feather`.
- Testing predictions prefer the packaged sample player-state artifact first.
- No `.env`, secret-directory, `/healthz`, or `/readyz` support exists in the current codebase.

## Before Extending The Project

Read [Tutorial Shortcuts vs Production](../tutorial-shortcuts-vs-production.md)
before treating the current auth, artifact, logging, or deployment behavior as a
production template.
