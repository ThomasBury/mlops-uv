# Getting Started

This guide gets the repo running locally with the commands that already exist in
the codebase.

## Prerequisites

- Python `>=3.10,<3.13`
- `uv`
- `just`
- Git
- Optional: Docker

## Install And Verify

```bash
uv sync
just test
```

That installs the locked environment and runs `uv run pytest tests`.

## Start The API

```bash
just serve
```

The development server listens on `http://127.0.0.1:8000`. Open
`http://127.0.0.1:8000/docs` for the generated OpenAPI UI.

The current API exposes these routes:

- `GET /`
- `POST /token`
- `GET /users/me/`
- `GET /users/me/items/`
- `POST /predict/`
- `GET /limit/`

## Generate Local Assets

The repository includes packaged sample assets, so `just serve` can start
without a local `data/` or `models/` directory. To exercise the project-root
asset path instead, run:

```bash
just data-check
just feature-state
just train
```

That validates `data/atp_data_production.feather`, writes player-state tables,
and exports a model to `models/model_*.joblib`.

## Build The Docs Site

```bash
just docs-serve
just docs-build
```

Those commands serve or build the Zensical site from `docs/`.

## Important Scope Notes

- The current code does not load `.env` files or secret-file directories.
- The current code does not implement `/healthz` or `/readyz`.
- Tutorial authentication uses an in-memory demo user and a static JWT signing key defined in code.

## Troubleshooting

- If `uv sync` fails, refresh the lockfile with `uv lock` and sync again.
- If port `8000` is busy, run `uv run fastapi dev src/acebet/app/main.py --host 127.0.0.1 --port 8001`.
- If local prediction fails for players that are missing from project data, rerun `just feature-state` or use `testing: true`.
