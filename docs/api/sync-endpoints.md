---
title: API Endpoints
description: Exact FastAPI routes, authentication, prediction schema, canonical response, and error behavior.
---

# API Endpoints

## Start And Authenticate

```bash
cp .env.example .env
just serve
```

The API listens on `http://127.0.0.1:8000`. The demo credentials default to
`johndoe` / `secret` and can be changed through
`ACEBET_DEMO_USERNAME` and `ACEBET_DEMO_PASSWORD`.

```bash
just token-json
just users-me
```

## Route Surface

| Method | Path | Auth | Behavior |
| --- | --- | --- | --- |
| GET | `/` | none | Welcome response |
| GET | `/healthz` | none | Process liveness |
| GET | `/readyz` | none | Startup-loaded asset readiness |
| GET | `/metrics` | none | Prometheus text; omitted from OpenAPI |
| POST | `/token` | none | Demo-user bearer token |
| GET | `/users/me` | bearer | Typed current-user identity |
| POST | `/predict/production` | bearer | Production asset prediction |
| POST | `/predict/testing` | bearer | Sample/testing assets outside production |

`/users/me/` is not a route and returns `404`; slash redirects are disabled.

Testing prediction is available only when `ACEBET_ENV` is `development` or
`test`.

## Prediction Request

```json
{
  "p1_name": "Fognini F.",
  "p2_name": "Jarry N.",
  "date": "2018-03-04",
  "series": "ATP250",
  "court": "Outdoor",
  "surface": "Hard",
  "round": "1st Round",
  "best_of": 3
}
```

All fields are required. Player names are trimmed, non-empty, case-sensitive,
and distinct. Enumerated values are exposed by OpenAPI. Extra fields, including
`atp`, `location`, `tournament`, and `testing`, are rejected.
Runtime mode comes from the route.

## Canonical Packaged-Bundle Response

The request above sent to `/predict/production` returns exactly:

```json
{
  "predicted_winner": "Fognini F.",
  "predicted_class": 1,
  "p1_win_probability": 0.768387902579221,
  "prediction_mode": "production",
  "model": {
    "kind": "elo_baseline",
    "version": "model_2026-08-02-12-59-07",
    "sha256": "a51b64ddddc094260ff41aa68beba9662bdb7769076130f55b1ed54a51a703c7"
  },
  "feature_freshness": {
    "p1_as_of_date": "2018-03-04",
    "p2_as_of_date": "2018-03-04",
    "p1_age_days": 0,
    "p2_age_days": 0
  }
}
```

The probability is the exact player-1 probability and is not rounded. Class
`1` selects player 1; class `0` selects player 2. Model identity comes from
validated schema-v2 metadata.

## Errors And Helpers

- `401`: missing/invalid token or credentials;
- `403`: testing route disabled in the current environment;
- `422`: native body validation or structured prediction-domain failures for
  equal/unknown players, unsupported context, future snapshots, or expected
  scoring failures;
- `429`: token or prediction rate limit;
- `503`: required model, identity, or player state was not loaded.

`just root`, `healthz`, `readyz`, `metrics`, `token-json`,
`users-me`, `predict-testing`, and `predict-production` target a running
API. `just manual` starts a temporary API and runs the authenticated flow.
