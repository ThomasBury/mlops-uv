# Modern MLOps and ML Engineering

At noon, what could we have known about two players before their match? Start with a hand calculation. Rebuild a named 2021 tennis dataset, train a probability model, compare it with Elo on later fixtures, and serve the same forecast offline, over HTTP and in a container. Use delayed results to investigate quality and decide whether to release the candidate.

[Start the course](course/README.md) for the milestone index, practice checkpoints, and ready-to-use prerequisites for the serving and registry labs.

The first lesson needs only a pencil. Use a fresh checkout for the practical work.

```bash
git clone https://github.com/ThomasBury/mlops-uv.git
cd mlops-uv
```

The course uses Python 3.12, uv and a Unix-style shell. The runtime lesson uses Docker Engine with Buildx or native Podman on Linux. M7 introduces `just` and GitHub Actions.

## Run the packaged predictor

AceBet ships an Elo model for the 2018 Brasil Open final between Fabio Fognini and Nicolás Jarry. With Python 3.12, uv, `just`, and curl installed, start the authenticated API from the checkout.

```bash
uv sync --locked --python 3.12
[ -f .env ] || cp .env.example .env
just serve
```

In a second terminal at the checkout root, run `just predict-production`. The response gives Fognini about a 77% win probability. Compare `model.sha256` with the digest in the [promotion record](src/acebet/data/promoted_model.json). The player snapshot dates describe the model's inputs and do not establish when those inputs first became available.

Licensed under Apache 2.0.
