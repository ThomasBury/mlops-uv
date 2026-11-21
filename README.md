# Classical MLOps and ML Engineering

![AceBet logo showing a tennis court and an orange ball](https://raw.githubusercontent.com/ThomasBury/mlops-uv/main/docs/images/acebet-mark.svg)

A modern, hands-on tutorial in classical MLOps and ML engineering. Across ten
lessons, you use a tennis match predictor to practice reproducible training,
evaluate models on later data, and serve, diagnose, and package a model-backed API.

AceBet is the Python application you work on throughout the tutorial.
**[Start the ten-lesson tutorial](https://mlops-uv-tutorial.bury-thomas.workers.dev/tutorial/).**

## Quickstart

Install Git, Python 3.12, `uv`, `just`, and `curl`. Run these commands in a
Unix-style shell.

```bash
git clone https://github.com/ThomasBury/mlops-uv.git
cd mlops-uv
uv sync --locked --python 3.12
[ -f .env ] || cp .env.example .env
just serve
```

On port 8000, the local API serves the packaged model. Leave it running and
request an example prediction from a second terminal in `mlops-uv`.

```bash
just predict-production
```

The packaged Elo model estimates a 77% chance of a Fognini win in this replay
of the 2018 Brasil Open final. The response's `model.sha256` is the fingerprint
of the model file that answered. Compare it with `model_sha256` in the
[committed promotion record](https://github.com/ThomasBury/mlops-uv/blob/main/src/acebet/data/promoted_model.json)
to verify that the API served the packaged model. See the
[full request and response](https://mlops-uv-tutorial.bury-thomas.workers.dev/api/sync-endpoints/).

The `just` commands are shortcuts for getting started. The tutorial walks
through the underlying workflow and explains what the shortcuts run. The
[setup guide](https://mlops-uv-tutorial.bury-thomas.workers.dev/getting-started/) and
[developer workflow guide](https://mlops-uv-tutorial.bury-thomas.workers.dev/uv/uv-workflows-for-mlops/)
cover the commands in more detail.

Licensed under Apache 2.0.
