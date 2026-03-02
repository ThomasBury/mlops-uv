# Training and Inference

This guide covers the standard training and inference workflow used in AceBet.

## Prerequisites

- Environment bootstrapped with `uv sync`
- Access to project data sources required by training and prediction code
- Familiarity with modules under `src/acebet/train` and `src/acebet/app`

## Training workflow

### 1. Ensure dependencies are installed

```bash
uv sync
```

### 2. Run the training script

```bash
uv run python -m acebet.train.train
```

### 3. Validate model artifact creation

```bash
ls -1 model_*.joblib | tail -n 1
```

## Inference workflow

### 1. Run local API for inference requests

```bash
uv run fastapi run src/acebet/app/main.py --host 0.0.0.0 --port 8000
```

### 2. Request an access token

```bash
curl -X POST http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=johndoe&password=secret"
```

### 3. Send a prediction request

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"p1_name":"Novak Djokovic","p2_name":"Rafael Nadal","date":"2016-01-01","testing":true}'
```

## Expected output

- Training command completes without exceptions and creates a `model_*.joblib` artifact.
- Token request returns an `access_token` and `token_type`.
- Inference API responds with HTTP 200 and JSON payload including `player_name`, `prob`, and `class_`.

## Troubleshooting

- **Training crashes on missing data**: verify the expected data file exists and matches schema.
- **Inference returns 422**: check request payload keys and data types against request model fields.
- **401 Unauthorized on prediction**: ensure the bearer token is valid and not expired.
