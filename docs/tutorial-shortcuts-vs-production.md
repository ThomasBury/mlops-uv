# Tutorial Boundaries

AceBet implements one compact, executable training-and-serving loop. The table
below is the single source for the line between demonstrated behavior and
production capabilities that are absent.

| Area | Implemented here | Not implemented |
| --- | --- | --- |
| Data provenance | Committed 44,617-row prepared Feather boundary; schema checks; deterministic re-derivation; 100-row transformation sample | Original source record, licensing record, governed ingestion, full raw production rebuild |
| Elo and snapshot lineage | Supplied Elo/rank fields; derived latest-player snapshot table; response snapshot dates and ages | Auditable Elo generation, point-in-time snapshot lineage, independent proof of causal availability |
| Evaluation and calibration | Strict date-grouped selection holdout; constant/Elo/LightGBM comparison; log-loss selection; Brier, ROC AUC, and accuracy diagnostics | Independent final test, multiple-window/backtest gate, reliability gate, post-hoc calibrator, proof that probabilities are calibrated |
| Artifacts and promotion | Schema-v2 metadata; dataset/model hashes; checksum-bound promotion pointer; no newest-file fallback; trusted-only joblib validation | Artifact signing or authenticity, external registry, approval workflow, cross-environment promotion |
| Serving and authentication | Startup-loaded assets; synchronous model work; exact player-1 probability; model identity and freshness; bearer token; one active demo user; bounded in-process rate limits | External identity provider, roles/scopes/tenants, distributed limiter, asynchronous model service, scaling or rollout controls |
| Observability | Structured logs and request IDs; bounded Prometheus labels; optional OpenTelemetry; local health/readiness | Central log/trace/metrics backend, dashboards, alerts, SLOs, retention and redaction governance |
| Deployment | Locked wheel/sdist builds; one non-root multi-stage inference image; packaged or explicit mounted assets; CI validation and Docker smoke; GitHub Pages docs deployment | Compose/orchestration, automated application deployment, registry publishing, scanning/signing, rollback, environment promotion |

Integrity checks and green validation establish that the repository agrees with
itself. They do not fill the missing provenance, causal, authenticity,
authorization, or deployment controls.
