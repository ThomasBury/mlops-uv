---
title: Serve one prediction
description: Send an authenticated request to your local bundle and verify the model identity in the response.
---

# 5. Serve one prediction { #tutorial-4-serving-validated-predictions }

<a id="why-this-exists"></a>

You have a promotion record and a table of player snapshots. Start the
application with both and verify that your model answers.

## Start the local API { #run-the-temporary-end-to-end-flow }

Run these commands from the repository root in **terminal A**.

```bash
[ -f .env ] || cp .env.example .env
just serve-with-assets var/tutorial/models var/tutorial/data
```

The recipe hands both directories to the application as settings.
At **startup**, before accepting requests, the app resolves the promotion
record, verifies and loads the model, then reads and validates the player
table. A configured directory must contain its required assets. Leave
terminal A running through lesson 7.

## Send a request { #implemented-prediction-path }

In **terminal B**, from the same repository root, save the match details
as JSON, a text format for named fields and values.

```bash
cat > var/tutorial/request.json <<'JSON'
{
  "p1_name": "Fognini F.",
  "p2_name": "Jarry N.",
  "date": "2018-03-04",
  "series": "ATP250",
  "court": "Indoor",
  "surface": "Clay",
  "round": "The Final",
  "best_of": 3
}
JSON

api=http://127.0.0.1:8000
(
  set -e
  response=var/tutorial/local-response.json
  rm -f "$response"
  pending="$(mktemp "$response.XXXXXX")"
  trap 'rm -f "$pending"' EXIT
  token="$(just --quiet token)"
  printf 'Authorization: Bearer %s\n' "$token" |
    curl -fsS "$api/predict/production" \
      -H @- \
      -H 'Content-Type: application/json' \
      --data-binary @var/tutorial/request.json \
      -o "$pending"
  mv "$pending" "$response"
  cat "$response"
)
```

The `token` recipe logs in with the demo credentials from `.env`. The
returned **bearer token** is a temporary credential. `-H @-` reads the
`Authorization` header from the pipe, keeping the token out of curl's
command-line arguments. The JSON body carries
the match details. HTTP is the protocol carrying the request and response,
and the `Content-Type` header tells the API how to read the body.
The `/predict/production` path selects the main model and player data even
when you run the app locally.

A new response is saved only after curl succeeds. If login or the request
fails, fix the reported error and rerun the block.

## Verify which model answered

Read `p1_win_probability` as the model's estimate for Fognini, the first player
in this request. Then compare the response's `model.version` to the promoted
filename without `.joblib`, and `model.sha256` to the full `model_sha256`
in the promotion record.

```bash
uv run python - <<'PY'
import json
from pathlib import Path

root = Path("var/tutorial")
served = json.loads((root / "local-response.json").read_text())["model"]
promoted = json.loads((root / "models/promoted_model.json").read_text())
assert served["version"] == Path(promoted["model_filename"]).stem
assert served["sha256"] == promoted["model_sha256"]
print("Served the promoted bundle:", served["version"])
PY
```

## Follow the scoring call { #why-prediction-handlers-are-synchronous }

FastAPI checks the submitted fields and supplies the signed-in user and
startup-loaded assets to the handler, the function for this endpoint. In
`src/acebet/app/routers/prediction.py`, follow `predict_match_outcome_production`
to `_predict_match_outcome`. It looks up the two snapshots and calls
`make_online_prediction` to assemble the model input and score it.
The response includes the identity of the loaded model.

The handler uses regular `def`. FastAPI runs it on a **worker thread**,
where blocking work such as assembling and scoring a pandas row can finish
while the **event loop** continues coordinating requests. The scoring
function runs inside that handler. If response times become a concern,
measure them first. Batch jobs or sustained heavy scoring may justify a
separate worker or model service. [FastAPI's explanation of threads and async](https://fastapi.tiangolo.com/async/#very-technical-details).

<a id="experiment-request-an-unknown-player"></a>

**Finish when** you can show which bundle answered and point to where the
request gets its player data. Keep terminal B open with the `api` address
set, and keep the request and local response files. Leave terminal A running
for [lesson 6 and its unknown-player exercise](./05-authentication-authorization-and-access-boundaries.md#exercise-the-authentication-flow).
