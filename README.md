# AceBet — Classical MLOps with uv

AceBet is a hands-on tutorial for Python users learning to train a tennis match
predictor and serve it through a local API.

**[Read the tutorial](https://github.com/ThomasBury/mlops-uv/blob/main/docs/tutorial/index.md)**

![](docs/images/acebet-mark.svg)

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

Leave the server running. In a second terminal, from the `mlops-uv` directory,
run the built-in example. It signs in with the demo credentials and requests
a prediction for Fognini against Jarry.

```bash
just predict-production
```

**Response excerpt**

```json
{
  "predicted_winner": "Fognini F.",
  "p1_win_probability": 0.768387902579221,
  "model": {"kind": "elo_baseline"}
}
```

`p1_win_probability` is the model’s estimate that Fognini, the first player in
the request, wins. `model.kind` tells you the Elo baseline answered. To verify
the exact saved model, compare `model.sha256` in the full response with
`model_sha256` in the [committed promotion record](src/acebet/data/promoted_model.json).

While the server is running, open the [local API reference at `/docs`](http://127.0.0.1:8000/docs).
For more options, see the [full setup guide](https://github.com/ThomasBury/mlops-uv/blob/main/docs/getting-started.md).

## What you’ll learn

- Compare candidate models on matches played after the training period.
- Verify which saved model answered a request by matching its checksum to the promotion record.
- Investigate a failed request using its request ID in the logs and the service’s metrics.

[Apache 2.0 License](LICENSE)
