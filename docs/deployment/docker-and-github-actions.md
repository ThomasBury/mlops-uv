# Deployment

This repository ships a Docker image path and a GitHub Pages docs deployment
path. It does not currently ship an automated application deployment workflow.

## Docker Image

Build the image:

```bash
docker build -t acebet .
```

Run it:

```bash
docker run --rm -p 8000:80 acebet
```

The Dockerfile copies the repository into `/app`, runs `uv sync --frozen
--no-cache`, and starts:

```bash
fastapi run src/acebet/app/main.py --host 0.0.0.0 --port 80
```

## Runtime Asset Behavior

The container follows the same asset resolution rules as local development:

- models from `/app/models`, then packaged data under `src/acebet/data`, then `/app`
- production player stats from `/app/data/latest_player_stats.feather`, then packaged production stats
- testing player stats from the packaged sample first

If you want the container to use locally generated project-root assets, mount
those directories:

```bash
docker run --rm -p 8000:80 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  acebet
```

## GitHub Actions In Scope

- `.github/workflows/main.yaml` runs the Python test suite.
- `.github/workflows/docs.yaml` builds and deploys the Zensical docs site to GitHub Pages.

If you want application deployment beyond local Docker runs, that still needs to
be designed and added explicitly.
