# Tutorial Boundaries

AceBet implements one compact, executable training-and-serving loop. The table
below is the single source for the line between demonstrated behavior and
production capabilities that are absent.

| Area | Implemented here | Not implemented |
| --- | --- | --- |
| Data provenance | Committed 44,617-row prepared Feather boundary; schema checks; deterministic re-derivation; 100-row transformation sample | Original source record, licensing record, governed ingestion, full raw production rebuild |
| Elo and snapshot lineage | Supplied Elo and rank; formula consistency check; [Elo timing probe](#elo-timing-probe); latest-player snapshots with dates, ages, and staleness warnings | Auditable upstream generation and point-in-time snapshot joins |
| Evaluation and calibration | Date-grouped selection holdout; separate later evaluation in the standard recipes; frozen LightGBM hyperparameters; log-loss selection; Brier, ROC AUC, and accuracy diagnostics | Inner hyperparameter search loop, multi-window rolling backtest gate, reliability gate, post-hoc calibrator, proof that probabilities are calibrated |
| Artifacts and promotion | Schema-v2 metadata; dataset, model, and production player-state digests; checksum-bound promotion pointer; no newest-file fallback; trusted-only joblib validation | Artifact signing or authenticity, external registry, approval workflow, cross-environment promotion |
| Serving and authentication | Startup-loaded assets; synchronous model work; exact player-1 probability; model identity and freshness; bearer token; one active demo user; bounded in-process rate limits | External identity provider, roles/scopes/tenants, distributed limiter, asynchronous model service, scaling or rollout controls |
| Observability | Structured logs and request IDs; bounded Prometheus labels; optional OpenTelemetry; local health/readiness | Central log/trace/metrics backend, dashboards, alerts, SLOs, retention and redaction governance |
| Deployment | Locked wheel/sdist builds; one non-root multi-stage inference image; packaged or explicit mounted assets; CI validation and Docker smoke; GitHub Pages docs deployment | Compose/orchestration, automated application deployment, registry publishing, scanning/signing, rollback, environment promotion |

Integrity checks and green validation establish that the repository agrees with
itself. They do not fill the missing provenance, causal, authenticity,
authorization, or deployment controls.

The post-selection dates are excluded from fitting and selection in the
current workflow. The date range used in the earlier LightGBM tuning pass
is unrecorded, so prior exposure to these dates is unknown.

## Elo timing probe

The prepared file contains Elo ratings and odds supplied by the upstream data.
The original rating code and point-in-time records are unavailable. Across
75,072 pairs of consecutive appearances by the same player, excluding dates
with multiple appearances by that player, Elo rose after 99.99% of wins and
fell after 99.91% of losses. The change aligned with the next match’s outcome
in 47.91% of wins and 50.21% of losses. This pattern is consistent with the
current row storing a rating from before its own match.

Across 32,663 matches with valid Pinnacle prices, Elo log loss was 0.6029
and the price-implied baseline was 0.5655 after normalizing inverse odds to
sum to one. Their probability correlation was 0.8716. These checks support
the pre-match interpretation. They leave the upstream generation and timing
of ranks and other inputs unverified.

```bash
uv run python -c 'from acebet.dataprep.dataprep import prepare_data, summarize_elo_causal_evidence; print(summarize_elo_causal_evidence(prepare_data()))'
```
