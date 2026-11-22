# Packaged predictor quickstart

Reproduce the 2018 Brasil Open final between Fabio Fognini and Nicolás Jarry using the Elo model shipped with AceBet. The response tells you which model answered and the dates of its player snapshots. For the 2021 project, start with the [pre-match forecast contract](tutorial/forecast-contract.md).

With Python 3.12, uv, just and curl installed, start the service from the checkout in a Unix-style shell.

```bash
uv sync --locked --python 3.12
[ -f .env ] || cp .env.example .env
just serve
```

Leave the local worker running. Open a second terminal at the checkout root to request the prediction.

```bash
just predict-production
```

The recipe obtains a bearer token and posts the players, date and match context to `/predict/production`. The following excerpt identifies the probability, model version and snapshot date.

```json
{
  "p1_win_probability": 0.768387902579221,
  "model": {"version": "model_2026-08-02-12-59-07"},
  "feature_freshness": {"p1_as_of_date": "2018-03-04"}
}
```

The model assigns Fognini about 77%. Compare `model.sha256` with `model_sha256` in `src/acebet/data/promoted_model.json`. Both snapshots are dated on match day. Their ages describe the inputs used for this replay. They cannot establish that those inputs were available before the first point.

[Authenticated request, complete response and errors](api/sync-endpoints.md) · [Model and data record](user-guide/model-artifacts-and-promotion.md) · [Download the 100-row notebook](assets/notebooks/01-starter-feature-and-training-flow.ipynb)
