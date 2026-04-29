# Tutorial Boundaries

AceBet is intentionally small enough to show one coherent training-and-serving
loop in a single repository. Several behaviors are tutorial shortcuts.

## Authentication

Current tutorial behavior:

- Uses a hardcoded JWT signing key.
- Uses an in-memory demo user store with one seeded account.
- Issues access tokens directly from the API process.

Production direction:

- Move credentials and token signing material out of source code.
- Use a real identity system or service-to-service auth flow.
- Add rotation, auditability, and per-environment secret management.

## Data And Model Artifacts

Current tutorial behavior:

- Reads the prepared dataset from the local filesystem.
- Writes trained models to `models/model_*.joblib`.
- Uses local feather files for online player-state lookups.
- Ships packaged fallback sample assets inside `src/acebet/data`.

Production direction:

- Store datasets and promoted model artifacts in governed remote storage or a registry.
- Track lineage and promotion state explicitly.
- Replace packaged fallback assets with environment-managed runtime data.

## Serving

Current tutorial behavior:

- Preloads model and player-state assets during FastAPI startup.
- Uses the `testing` flag only to switch preferred runtime assets.
- Exposes `/`, `/token`, `/users/me/`, `/users/me/items/`, `/predict/`, and `/limit/`.
- Does not expose `/healthz` or `/readyz`.

Production direction:

- Add explicit liveness and readiness semantics if the deployment platform needs them.
- Separate tutorial auth and prediction concerns from platform concerns such as proxies, scaling, and rollout safety.

## Observability

Current tutorial behavior:

- Logs to `info.log`.
- Captures a request-body excerpt for write requests.
- Does not emit structured metrics, traces, or request IDs.

Production direction:

- Use structured logging and centralized log shipping.
- Add metrics for latency, failures, rate limiting, and model-serving outcomes.
- Add trace correlation across API, storage, and training systems.

## Training And Promotion

Current tutorial behavior:

- Runs training locally from `just train`.
- Uses a time-series split and exports a local joblib artifact.
- Optionally logs to a local MLflow tracking setup.

Production direction:

- Run training in scheduled or event-driven pipelines.
- Track data, code, and artifact versions together.
- Gate promotion on stronger validation and explicit release controls.

## Automation

Current tutorial behavior:

- Tests run in GitHub Actions.
- Docs build and deploy through a separate GitHub Pages workflow.
- Docker packaging is documented and runnable locally, but not automatically deployed.

Production direction:

- Add container scanning, artifact signing, deployment workflows, and rollback procedures when the project moves beyond tutorial scope.
