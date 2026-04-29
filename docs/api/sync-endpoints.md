# API Endpoints

AceBet uses FastAPI route handlers backed by a small OAuth2 password flow and a
preloaded model/player-state runtime.

## Start The API

```bash
just serve
```

The current codebase does not provide `/healthz` or `/readyz`.

## Authentication

The demo account is fixed in code:

- username: `johndoe`
- password: `secret`

Get a token:

```bash
just token-json
```

Or with raw HTTP:

```bash
curl -X POST http://127.0.0.1:8000/token \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=johndoe&password=secret'
```

## Current Route Surface

| Method | Path | Auth | Behavior |
| --- | --- | --- | --- |
| `GET` | `/` | none | Returns the welcome message. |
| `POST` | `/token` | none | Issues a bearer token for the demo user. |
| `GET` | `/users/me/` | bearer token | Returns the authenticated user profile. |
| `GET` | `/users/me/items/` | bearer token | Returns the demo item list for that user. |
| `POST` | `/predict/` | bearer token | Builds an online feature row and scores it with the loaded model. |
| `GET` | `/limit/` | none | Demo rate-limited route. |

## Prediction Requests

Minimal request body:

```json
{
  "p1_name": "Fognini F.",
  "p2_name": "Jarry N.",
  "date": "2018-03-04",
  "testing": false
}
```

Optional fields are `atp`, `location`, `tournament`, `series`, `court`,
`surface`, `round`, and `best_of`.

`testing` does not mean "fake bet" versus "real bet". It only changes asset
selection:

- `testing: false` prefers project-root production player stats.
- `testing: true` prefers the packaged sample player stats first.

The model lookup order is unchanged in both modes.

## Existing Helper Commands

```bash
just root
just users-me
just users-me-items
just predict-testing
just predict-production
just limit
just manual
```

## Troubleshooting

- A `401` from `/users/me/`, `/users/me/items/`, or `/predict/` means the bearer token is missing or invalid.
- A `400` from `/predict/` usually means player stats are missing for one of the requested players.
- A `429` from `/limit/` means the route-specific limiter window has been exceeded.
