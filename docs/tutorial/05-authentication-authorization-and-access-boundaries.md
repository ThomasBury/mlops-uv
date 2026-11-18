---
title: Explain a rejected request
description: Tell when the caller needs a valid token and when the model needs player data.
---

# 6. Explain a rejected request { #tutorial-5-authentication-and-caller-identity }

<a id="why-this-exists"></a>

Keep the server running in terminal A. In terminal B, send two requests that
change one thing at a time from lesson 5. An HTTP status code tells the
caller how the server handled the request. The response body can narrow down
the reason.

## Omit the token { #experiment-omit-the-token }

```bash
curl -sS -w '\nHTTP %{http_code}\n' "$api/predict/production" \
  -H 'Content-Type: application/json' \
  --data-binary @var/tutorial/request.json
```

Expect `401` and `Not authenticated`. The authentication check requires a
valid token for the configured active user. The caller can log in again and
send the returned token.

## Ask about an unknown player { #exercise-the-authentication-flow }

```bash
token="$(just --quiet token)"
sed 's/Fognini F[.]/Tutorial Unknown Player/' var/tutorial/request.json |
  curl -sS -w '\nHTTP %{http_code}\n' "$api/predict/production" \
    -H "Authorization: Bearer $token" \
    -H 'Content-Type: application/json' \
    --data-binary @-
unset token
```

Expect `422` with `detail.code` set to `unknown_players`. Authentication
passes. The request stops at the snapshot lookup, and the body names the
missing player. If the spelling is correct, supplying that player's state
is a job for the service operator.

## Where the credentials come from { #implemented-security-boundary }

Open `.env.example` and find `ACEBET_DEMO_USERNAME` and
`ACEBET_DEMO_PASSWORD`, the values used to request the token in this exercise.
The service checks the token's signature using a signing key. The example
opts into a built-in development key for this loopback-only exercise.
Production settings require an explicitly supplied password and a real
signing key. Provide those through deployment configuration or secret files.
See [API Endpoints](../api/sync-endpoints.md) and
[Tutorial Boundaries](../tutorial-shortcuts-vs-production.md) for access and
rate-limit details.

**Finish when** you can tell a caller what to do for each status and explain
which part of the request path rejected it. Keep both terminals and the
saved request for [lesson 7](./06-observability-logging-metrics-and-traces.md).
