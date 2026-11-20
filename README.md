# Classical MLOps with `uv`

Build, evaluate, and serve a tennis match predictor in ten hands-on lessons.
AceBet is the Python application you work on throughout the tutorial.

**[Read the web documentation](https://thomasbury.github.io/mlops-uv/)** or
[start the tutorial](https://thomasbury.github.io/mlops-uv/tutorial/).

## What you’ll learn

- Recreate a Python environment from a lockfile with `uv`.
- Build model inputs, compare models on later matches, and choose which saved model to serve.
- Serve an authenticated prediction, verify which model answered, and investigate requests using logs, metrics, and traces.
- Package the application, reproduce the runtime in Docker, and decide what CI should verify before release.

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
of the 2018 Brasil Open final. See the [full request and response](https://thomasbury.github.io/mlops-uv/api/sync-endpoints/).

The `just` commands are shortcuts for getting started. The tutorial walks
through the underlying workflow and explains what the shortcuts run. The
[setup guide](https://thomasbury.github.io/mlops-uv/getting-started/) and
[developer workflow guide](https://thomasbury.github.io/mlops-uv/uv/uv-workflows-for-mlops/)
cover the commands in more detail.

## Tutorial chapters

1. [Recreate the working environment](https://thomasbury.github.io/mlops-uv/tutorial/00-recreate-the-working-environment/)
2. [Establish what a prediction needs](https://thomasbury.github.io/mlops-uv/tutorial/01-notebook-to-data-contract/)
3. [Make a training decision reproducible](https://thomasbury.github.io/mlops-uv/tutorial/02-feature-contract-training-and-export/)
4. [Choose the bundle the service may load](https://thomasbury.github.io/mlops-uv/tutorial/03-reproducibility-versioning-and-dependencies/)
5. [Serve one prediction](https://thomasbury.github.io/mlops-uv/tutorial/04-serving-async-and-monitoring/)
6. [Explain a rejected request](https://thomasbury.github.io/mlops-uv/tutorial/05-authentication-authorization-and-access-boundaries/)
7. [Investigate a running service](https://thomasbury.github.io/mlops-uv/tutorial/06-observability-logging-metrics-and-traces/)
8. [Hand the application to another Python user](https://thomasbury.github.io/mlops-uv/tutorial/08-hand-the-application-to-another-python-user/)
9. [Reproduce the service runtime](https://thomasbury.github.io/mlops-uv/tutorial/09-reproduce-the-service-runtime/)
10. [Decide what to verify before release](https://thomasbury.github.io/mlops-uv/tutorial/07-validation-ci-deployment-and-boundaries/)

Licensed under Apache 2.0.
