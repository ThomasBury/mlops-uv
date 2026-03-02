# Getting Started

This project uses **Zensical** to build and preview documentation from Markdown files under `docs/`.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Install tooling

Run these commands from the repository root:

```bash
uv sync --dev
uv add --dev zensical
```

## Bootstrap docs (if needed)

```bash
zensical new .
```

## Build docs locally

```bash
zensical build --clean
```

## Preview docs locally

```bash
zensical serve
```

## Contribution flow for docs updates

1. Create a branch for your documentation update.
2. Add or edit Markdown files in `docs/`.
3. Run a local docs build:
   ```bash
   zensical build --clean
   ```
4. Run local preview and validate rendering:
   ```bash
   zensical serve
   ```
5. Commit your changes and open a pull request.

On pull requests, GitHub Actions builds docs automatically. On pushes to `main`, docs are published to GitHub Pages.
