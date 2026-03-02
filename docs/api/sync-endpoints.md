# API: Sync Endpoints

AceBet uses FastAPI with synchronous route functions (`def`) for core endpoints.

## Prerequisites

- Environment prepared with `uv sync`
- API server running locally
- Optional: valid OAuth2 credentials for protected routes

## Start the API

```bash
uv run fastapi run src/acebet/app/main.py --host 0.0.0.0 --port 8000
```

## Common sync endpoint calls

### 1. Health/home endpoint

```bash
curl http://localhost:8000/
```

### 2. Rate-limited endpoint

```bash
curl "http://localhost:8000/limit/?user_id=test-user"
```

### 3. Get token (password flow)

```bash
curl -X POST http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=johndoe&password=secret"
```

### 4. Protected prediction endpoint

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=johndoe&password=secret" | jq -r '.access_token')

curl -X POST http://localhost:8000/predict/ \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"p1_name":"Novak Djokovic","p2_name":"Rafael Nadal","date":"2016-01-01","testing":true}'
```

## Expected output

- `GET /` returns `{"message":"Welcome to the AceBet API!"}`.
- `GET /limit/` returns success until the per-minute limit is exceeded.
- `POST /token` returns an `access_token` and `token_type`.
- `POST /predict/` returns JSON with `player_name`, `prob`, and `class_`.

## Troubleshooting

- **401 Unauthorized on `/predict/`**: ensure `Authorization: Bearer <token>` is present and token is fresh.
- **429 Too Many Requests on `/limit/`**: wait for the limiter window to reset.
- **`jq: command not found`**: parse token manually or install `jq`.
