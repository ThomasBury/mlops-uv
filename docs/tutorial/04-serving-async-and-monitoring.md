---
title: Tutorial 4 - Serving Validated Predictions
description: Follow the implemented synchronous prediction path, validated model identity, exact probabilities, and snapshot freshness.
---

# Tutorial 4: Serving Validated Predictions

The serving contract is deliberately narrow: validate runtime assets at
startup, build one feature row from two snapshots and explicit match context,
score the selected model, and return the exact evidence a caller needs.

## Why This Exists

**Serving** turns a selected bundle into a request-time prediction. The small
end-to-end command proves the API loads validated assets, authenticates a
caller, and returns the evidence needed to interpret one prediction.

## Run The Temporary End-To-End Flow

```bash
[ -f .env ] || cp .env.example .env
just manual
```

The first command preserves any existing `.env` and creates the local tutorial
auth configuration only when it is missing.

The recipe starts `fastapi run` on a free local port, waits for readiness,
obtains a token, exercises both prediction modes, and removes the temporary
process. It does not regenerate data or model artifacts.

## Implemented Prediction Path

For either prediction route the API:

1. uses the mode-specific startup-loaded model and player-state table;
2. resolves both snapshots and calculates their calendar-day ages;
3. builds exactly the shared 17-column feature contract;
4. scores the validated promoted estimator;
5. returns class, winner, exact player-1 probability, mode, model identity, and
   freshness.

Class `1` means player 1 wins; class `0` means player 2 wins. The response
does not round or convert the player-1 probability. For the canonical request,
the packaged Elo bundle returns `0.768387902579221`.

Model identity is not inferred from a directory name. It comes from the same
validated schema-v2 metadata as the loaded estimator and includes kind,
version, and the complete model SHA-256.

Snapshot dates and ages disclose how old each player's rank and Elo state was
at the requested match date. Unknown players and snapshots newer than the match
date produce structured `422` responses. Missing runtime assets produce
`503`.

## Why Prediction Handlers Are Synchronous

Pandas feature assembly, joblib-backed model work, and password hashing are
blocking operations. Prediction routes and their blocking dependencies are
regular `def` functions so FastAPI runs them in its worker threadpool.
Application lifespan and request middleware remain asynchronous where they
coordinate startup, shutdown, and request flow.

The service does not implement a task queue, asynchronous model server, batch
endpoint, GPU scheduler, or autoscaling policy. Those boundaries are summarized
once in [Tutorial Boundaries](../tutorial-shortcuts-vs-production.md).

!!! info "Shortcut"

    **Why it is enough here:** one synchronous API keeps blocking model work
    correct and visible. **Add it when…** throughput, batch work, GPUs, or
    independent scaling require a queue or model service. See
    [Tutorial Boundaries](../tutorial-shortcuts-vs-production.md).

## Experiment: Request An Unknown Player

**Prerequisites:** complete Getting Started and keep `just serve` running on
`127.0.0.1:8000` in another terminal. Run these commands from the repository root.
**Prediction:** valid authentication cannot supply missing player state; the
request returns `422` with `unknown_players`.

```bash
token="$(just --quiet token)"
curl -sS -w '\nHTTP %{http_code}\n' \
  http://127.0.0.1:8000/predict/production \
  -H "Authorization: Bearer ${token}" \
  -H 'Content-Type: application/json' \
  -d '{"p1_name":"Tutorial Unknown Player","p2_name":"Jarry N.","date":"2018-03-04","series":"ATP250","court":"Outdoor","surface":"Hard","round":"1st Round","best_of":3}'
unset token
```

Expected output (the error body remains visible):

```text
{"detail":{"code":"unknown_players","message":"Player snapshots are unavailable for one or more requested players.","context":{"players":["Tutorial Unknown Player"]}}}
HTTP 422
```

**Explain:** why does this fail at snapshot resolution rather than estimator
scoring? What evidence would adding this player require?
**Cleanup:** the token variable is unset; no files change. Keep the API running
for the next chapters, then stop it with Ctrl-C.

Continue with
[Authentication And Caller Identity](./05-authentication-authorization-and-access-boundaries.md).
