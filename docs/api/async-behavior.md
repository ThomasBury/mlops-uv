# API: Async Behavior

Even with synchronous route handlers, AceBet uses asynchronous behavior through FastAPI/Starlette middleware and request handling.

## Prerequisites

- Environment installed with `uv sync`
- API server running locally
- Basic understanding of FastAPI middleware lifecycle

## Run the API and inspect async behavior

### 1. Start server

```bash
uv run fastapi run src/acebet/app/main.py --host 0.0.0.0 --port 8000
```

### 2. Generate requests to trigger middleware logging

```bash
curl http://localhost:8000/
curl "http://localhost:8000/limit/?user_id=async-check"
```

### 3. Review generated logs

```bash
tail -n 20 info.log
```

## What is async here?

- Request middleware (`user_logging_middleware`) is asynchronous and awaits request/response processing.
- Request body handling uses async receive channel manipulation.
- The middleware collects streamed response chunks asynchronously before writing logs.

## Expected output

- Server continues handling requests without blocking between middleware stages.
- `info.log` contains request and response payloads.
- No middleware-related stack traces during normal traffic.

## Troubleshooting

- **`info.log` not created**: ensure the process has write permissions in the working directory.
- **Incomplete logs**: verify requests reach the app and response body iteration completes.
- **High latency during heavy logging**: reduce log verbosity or add async-safe log sinks for production.
