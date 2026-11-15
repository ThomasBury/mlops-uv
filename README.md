# Classical MLOps With UV

Tutorial site: <https://thomasbury.github.io/mlops-uv/>

[![Tests](https://github.com/ThomasBury/mlops-uv/actions/workflows/main.yaml/badge.svg)](https://github.com/ThomasBury/mlops-uv/actions/workflows/main.yaml)
[![Documentation](https://github.com/ThomasBury/mlops-uv/actions/workflows/docs.yaml/badge.svg)](https://github.com/ThomasBury/mlops-uv/actions/workflows/docs.yaml)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

`acebet` is a compact MLOps tutorial for a tabular tennis match predictor. The
repo covers prepared-data validation, feature assembly, model training, FastAPI
serving, tests, Docker packaging, and optional MLflow tracking without adding
infrastructure that the code does not actually implement.

The canonical documentation source lives in `docs/` and is built by Zensical
using `zensical.toml`. For the full walkthrough, start with
[`docs/index.md`](docs/index.md).

## Quickstart

```bash
uv sync --locked
cp .env.example .env
just serve
```

Then open `http://127.0.0.1:8000/docs` for OpenAPI.

To preview the authored docs site separately, run `just docs-serve` and open
`http://127.0.0.1:8002`.

## GitHub Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ThomasBury/mlops-uv&type=Date)](https://star-history.com/#ThomasBury/mlops-uv&Date)
