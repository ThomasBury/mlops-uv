---
title: Tutorial 5 - Authentication And Caller Identity
description: Obtain a bearer token, protect prediction routes, and reuse the authenticated caller identity.
---

# Tutorial 5: Authentication And Caller Identity

AceBet places one concrete access boundary in front of prediction: callers must
obtain a token and resolve to the active configured demo user.

## Why This Exists

**Authentication** establishes who is calling; **authorization** decides what
that caller may do. The commands below prove the first boundary for one demo
user, then reuse that identity for prediction.

## Exercise The Authentication Flow

Start the API after completing Getting Started, then run:

```bash
just token-json
just users-me
just predict-production
```

`POST /token` accepts the OAuth2 password-style form, checks the configured
demo credentials, and returns a signed bearer token. Both prediction routes
require that token through the reusable current-active-user dependency.

`GET /users/me` is retained here as a focused demonstration that decoded
caller identity can be injected into route logic and returned as a typed
profile. It is also documented in the exact API reference; it is not part of
the model-serving narrative.

## Implemented Security Boundary

- demo username and password are settings-backed;
- JWT signing material comes from environment, `.env`, or a mounted secret
  file;
- the insecure fallback requires an explicit development-only opt-in;
- `/token` has a client-address rate limit;
- production and testing predictions share a user-keyed rate limit;
- disabled users are rejected.

This is authentication with a minimal active-user check. It is not a complete
authorization system: there are no roles, scopes, tenant isolation, external
identity provider, distributed limiter, credential store, or key-rotation
workflow.

Operational probes and metrics remain unauthenticated so local health checks
and scraping can work. Exact routes, status codes, and request shapes live in
[API Endpoints](../api/sync-endpoints.md); broader gaps live in
[Tutorial Boundaries](../tutorial-shortcuts-vs-production.md).

!!! info "Shortcut"

    **Why it is enough here:** one active demo user makes the access boundary
    testable. **Add it when…** callers need distinct permissions or tenancy;
    then add an identity provider, roles/scopes, and durable rate limiting. See
    [Tutorial Boundaries](../tutorial-shortcuts-vs-production.md).

## Experiment: Omit The Token

**Prerequisites:** keep the local API from Getting Started running on
`127.0.0.1:8000`; this command needs curl but no credentials.
**Prediction:** the protected identity endpoint rejects the request with `401`.

```bash
curl -sS -w '\nHTTP %{http_code}\n' http://127.0.0.1:8000/users/me
```

Expected output:

```text
{"detail":"Not authenticated"}
HTTP 401
```

**Explain:** why can `/healthz` succeed while this request fails? What would
this experiment need to test authorization beyond one active demo user?
**Cleanup:** no files or shell variables are created. Stop the API with Ctrl-C
when the serving experiments are finished.

Continue with
[Implemented Observability](./06-observability-logging-metrics-and-traces.md).
