# Rebuild AceBet around a trustworthy match forecast

## Task and review boundary

Design a local, continuous project for learners who already know Python, statistics, and machine learning. AceBet gives a reader comparing two named tennis players the probability that the first player wins, using information available at a declared cutoff before the match. Success means a learner can take an unfamiliar prediction, identify which model answered and what it knew, diagnose a plausible error, and demonstrate a repair.

The curriculum runs from the [M1 forecast contract](docs/tutorial/forecast-contract.md), the [M2 installed fixture replay](docs/tutorial/reproduce-the-fixture.md), the [M3 match history pipeline](docs/tutorial/rebuild-match-history.md), the [M4 historical model comparison](docs/tutorial/evaluate-a-forecast.md), the [M5 frozen forecast service](docs/tutorial/serve-the-frozen-forecast.md), and the [M6 monitoring and rollback drill](docs/tutorial/diagnose-and-roll-back-a-forecast.md) to the [M7 automation and audit exercise](docs/tutorial/automate-and-audit-a-forecast.md). The M7 human walkthrough is pending. The rebuild may replace the current API, artifacts, commands, and chapter order. Existing code earns its place by passing the new checks.

M3 derives player history from a pinned, redistributable extract of Valuebetennis results. The existing 100-row raw sample demonstrates a transformation. The 44,617-row prepared history has incomplete upstream provenance, including the timing of supplied Elo ratings. [Current data and Elo boundaries](docs/tutorial-shortcuts-vs-production.md)

## Learning path

Work through M1–M7 in order. Each milestone uses the previous one's output. Carry one fixture through the contract, historical replay, evaluation, service response, and rollback. Run the core replay on an ordinary laptop in seconds to a few minutes, without paid infrastructure. Each lesson starts with a decision, shows a failure, introduces the smallest adequate remedy, and gives the learner a command, an expected observation, and a reason to revisit the design.

### M1 · Define the ML problem

- [x] **Objective and prerequisite.** Given basic probability and ML knowledge, specify who uses a pre-match probability, what a row represents, which matches qualify, how players are ordered, the cutoff for inputs, and when the outcome is observable.
- **Exercise.** Draw one fixture's timeline. Demonstrate how a source that places the winner first can reveal the label through player position alone. Define outcome-independent ordering, treatment of walkovers and retirements, and the status of a fixture still awaiting its result.
- **Evidence.** Give a second reader a completed match, a walkover, and an unresolved fixture. Using the contract alone, they should agree on eligibility, allowed inputs, and whether each forecast can be scored. Record a simple baseline, log loss for probability quality, calibration as a diagnostic, and separate software and operational success criteria.
- **Handoff.** Use synthetic fixture `F01` and its hand-worked `0.50` forecast in M2. The lesson supplies an answer key for independent review. Agreement with a second reader is pending.
- **Trade-off.** Estimating a win probability is sufficient for this reader. A betting decision would also require prices, costs, and a decision rule.

### M2 · Build a reproducible Python project

- [x] **Objective and prerequisite.** Turn M1's hand-worked forecast into a typed package function with a runnable entry point. Explain the roles of an exploratory notebook, a script, an importable package, and a service.
- **Exercise.** Declare project metadata and dependencies in `pyproject.toml`, use `uv` to record the resolved dependencies in `uv.lock`, and make configuration explicit. Use type hints, Ruff, a pytest unit test for the calculation, and an integration test through the installed entry point. Try importing from outside the checkout to expose reliance on the working directory.
- **Evidence.** From a fresh clone, recreate the environment with the recorded Python version and `uv sync --locked`, run lint and the full suite, and reproduce the fixture result without local secrets or an existing model.
- **Handoff.** `acebet-replay` prints F01, its noon UTC cutoff, the ordered player IDs, and the `0.50` baseline. Carry that fixture and cutoff into M3's historical replay.
- **Trade-off.** A lockfile identifies software dependencies. The local environment can still change. M3 identifies the data separately, and M5 identifies the built runtime image by digest.

### M3 · Build a trustworthy data and feature pipeline

- [x] **Objective and prerequisite.** Use M1's cutoff and M2's package to answer two questions independently. Can we reconstruct the records, and can we reconstruct what was knowable about a player at the cutoff?
- **Exercise 1 — reconstruct the data.** Accept a real source only after verifying redistribution terms, revision, coverage, stable match and player identities, and date semantics. Preserve the raw snapshot, attribution, and checksum. Implement ingestion and pure transformations as explicit Python stages writing canonical Parquet. Check schema, required values, uniqueness, and ranges. Ingest twice, inject a conflicting duplicate, and preserve the original snapshot when recording a correction.
- **Exercise 2 — reconstruct player history.** Hand-calculate Elo for a few matches, including a new player. Replay fixtures with a shared as-of lookup. Deliberately update a rating with the current result before forecasting, observe the optimistic score on development data, then repair the update order. Use completed earlier periods when the source cannot establish within-period ordering. Document any assumed delay between event and availability.
- **Evidence.** Rebuild identical canonical content from the recorded source. Demonstrate that a future record, a late correction, and a change to the current winner cannot alter earlier feature values. Trace the worked fixture to contributing records. Change an input or transformation and show why an old cached result is stale.
- **Handoff.** The [runnable lesson](docs/tutorial/rebuild-match-history.md) rebuilds 1,856 completed matches and replays F01 from its noon cutoff. M4 can use the canonical Parquet and the shared as-of Elo replay. The leakage comparison uses 1,487 matches before September 1, 2021. The remaining 369 matches are available for final evaluation.
- **Trade-off.** Use Python checks and Parquet for the core exercise. Discuss SQL and DuckDB when file inspection or joins become cumbersome, Pandera when multiple producers need one shared schema, and DVC when distributing and retrieving versioned inputs becomes costly. Cheap stages can be recomputed.

### M4 · Train, evaluate, and track models

- [x] **Objective and prerequisite.** Use M3's historical replay to decide whether a candidate improves the forecast. Define expanding chronological development folds, a later untouched final period, and a rule for choosing the candidate before examining final scores.
- **Exercise.** Compare training prevalence, historical Elo, and logistic regression on the same fixtures. Fit numeric and categorical preprocessing within each training fold using a scikit-learn `Pipeline` and `ColumnTransformer`. Try an unseen surface. Contrast the leaky rating with the repaired version on development folds. Inspect how a random split lets a model learn from later matches.
- **Experiment record.** After two meaningful comparisons, introduce local MLflow. Record dataset digest, feature and code revision, dependencies, configuration, seeds, split membership, parameters, metrics, and artifact identities. Run a small seeded `RandomizedSearchCV` over the temporal development folds. Freeze the selection decision and evaluate the selected candidate and prespecified baselines on the final period.
- **Evidence.** Report log loss, a calibration summary, and the uncertainty of paired loss differences using time blocks. Explain the block choice and the limits of a short history. Rerun training from recorded inputs. Compare fixture predictions and metrics against tolerances declared before the rerun. Include an inconclusive result or baseline win when supported. [scikit-learn guidance on preprocessing and leakage](https://scikit-learn.org/stable/common_pitfalls.html) · [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- **Handoff.** The [runnable lesson](docs/tutorial/evaluate-a-forecast.md) freezes the logistic pipeline and replays F01. On 369 final matches, log loss was 0.62479 for the candidate and 0.62609 for Elo. The paired interval includes zero. M5 can take the frozen bundle, its canonical history, and the F01 trace into the serving exercise.
- **Trade-off.** Returning players legitimately occur across periods. A player-disjoint split answers a cold-start question. Consider LightGBM after identifying a limitation of the simple model, and Optuna when a bounded search becomes expensive.

### M5 · Serve and package the model

- [x] **Objective and prerequisite.** Take M4's frozen model, fitted preprocessing, feature contract, and identified historical state from a local batch prediction to an HTTP response.
- **Exercise.** Expose the same prediction path through FastAPI with Pydantic validation and Uvicorn. Load and verify compatible assets at startup. Distinguish process liveness from readiness to predict. Build a minimal multi-stage Docker image with a non-root runtime. Replay the fixture through the local model and the container.
- **Evidence.** Compare probabilities and identities for the same players and cutoff. Exercise invalid input, corrupt or mismatched assets, and concurrent requests. Deliberately block a request handler and measure the effect on another request. Explain the event loop, threads, the GIL, and worker processes using the observed behavior.
- **Handoff.** The [runnable lesson](docs/tutorial/serve-the-frozen-forecast.md) replays F01 offline, over HTTP, and in a non-root image with no mounts. Keep the matching model and feature-state checksums, the image ID, and the request for M6. On a local one-worker run, the overlapping liveness request took 802–806 ms with a blocked event loop and 5–6 ms with the sleep in a thread.
- **Trade-off.** Batch scoring serves a known list of fixtures cheaply. Online inference adds validation, startup, concurrency, and latency obligations. Use measurements to justify extra workers.

### M6 · Observe and maintain the system

- [x] **Objective and prerequisite.** Given M5's service, determine what failed, which forecasts were affected, and whether enough labels have arrived to assess quality.
- **Exercise.** Use structured Python logging, request metrics, and one trace connecting feature lookup and inference. Introduce Prometheus metric and OpenTelemetry trace concepts through that request. Observe throughput, errors, latency, concurrency, and resource use. Separately track input and prediction distributions. Join delayed results to forecast IDs and report label coverage, log loss, and calibration by forecast period.
- **Evidence.** Inject invalid requests, latency, and a shifted input population separately and recover the expected signal for each. Delay labels and inspect which metrics remain interpretable. Reject an inferior candidate at the promotion gate. Separately rehearse rollback after a deployment, restoring both model and feature state, then replay the fixture.
- **Handoff.** The [runnable lesson](docs/tutorial/diagnose-and-roll-back-a-forecast.md) traces F01, scores an archived forecast when its result arrives, and restores both model and feature state. Keep the test output, the gate decision, and both deployment identities for M7. In the archived comparison, the prevalence candidate was rejected with a paired log-loss difference of +0.0682 and a 95% interval from +0.0392 to +0.0877.
- **Trade-off.** Input drift can justify investigation. Assessing a change in the relationship between inputs and outcomes requires mature labels and enough observations. Keep retraining, acceptance, and activation as explicit decisions.

### M7 · Automate and audit the validated workflow

- [ ] **Objective and prerequisite.** After executing M1–M6 manually, automate the repeated steps and demonstrate that an independent learner can challenge a plausible result.
- **Exercise.** Expose preparation, training, evaluation, testing, and serving through a `justfile`. Use a small GitHub Actions workflow to run the checks and replay from a clean checkout. Demonstrate a scheduled candidate run with its validation record and an explicit promotion step.
- **Evidence.** Hand a learner an unfamiliar AceBet pipeline with an undisclosed defect. Ask them to identify the affected claim, show the failure, repair it, and provide a regression check. Record a human walkthrough independently of CI. A reviewer can locate the command and evidence for every row below.
- **Handoff.** The [runnable lesson](docs/tutorial/automate-and-audit-a-forecast.md) indexes each check and shows how to inspect the scheduled candidate's `validation.json`, rerun the Elo gate, and attempt explicit local promotion. The archived comparison is inconclusive against Elo. Hosted scheduling and an independent human walkthrough remain to be observed.
- **Trade-off.** A local promotion record suffices for the exercise. Discuss a registry when environments must coordinate approvals, an orchestrator when retries and backfills cross dependent jobs, and a feature store when several systems must share historical and online feature semantics.

## Independent verification matrix

For each row, the implementation supplies a runnable command or short exercise, its expected observation, and the evidence a reviewer can inspect. A milestone closes when its evidence can be reproduced independently.

| Claim | Reviewer action and passing observation | Milestone |
| --- | --- | --- |
| The forecast has a defined target | Classify the worked fixtures before opening the answer key. Identify which inputs qualify, recover both possible labels with fixed player order, and reproduce the baseline loss on the two scorable fixtures. | M1 |
| Environment is reproducible | Start from a fresh clone and empty environment. Run `uv sync --locked` with the recorded Python version, then Ruff and the full pytest suite. Verify the installed entry point works outside the checkout. | M2, M7 |
| Data is auditable | Verify the raw revision and checksum, rebuild from it, and compare schema, row identities, labels, and a stable digest of canonical content. | M3 |
| Ingestion is idempotent | Ingest twice into the same destination. Match identities, counts, and the canonical digest agree after both runs. | M3 |
| Future information is excluded | Inject a future-dated feature and a correction that becomes available after the cutoff. Earlier features and predictions stay identical. The deliberately leaky implementation fails the hand-worked example. | M3, M4 |
| Validation simulates deployment | Inspect exported membership, date boundaries, and label maturity. Each match and its duplicates belong to one partition per split. Explain player overlap. Verify tuning and preprocessing never fit on the final period. | M4 |
| Model value is measured | Score the frozen candidate and baselines on identical final fixtures. Report paired loss differences and time-block uncertainty. Claim improvement only to the extent supported. | M4 |
| Training is reproducible | Rebuild and retrain from recorded data, code, dependencies, configuration, and seeds. Compare predictions and metrics within predefined tolerances. Require identical artifact hashes only where byte identity is promised. | M4 |
| Serving matches training | Replay the same ordered players, context, and cutoff offline and over HTTP. Check probability agreement within tolerance and match the model and feature-state identities. | M5 |
| Deployment is reliable | Build and launch without host model directories. Verify non-root execution, readiness, invalid-request responses, concurrent predictions, and failure to become ready with an incompatible bundle. | M5 |
| Monitoring works | Inject errors, slow inference, and a shifted population independently. Inspect the corresponding logs, metrics, and trace. Withhold recent labels and verify that performance reporting exposes coverage. | M6 |
| Rollback works | Verify rejection of an inferior candidate leaves the active deployment intact. Activate another version locally, restore the known-good model and feature state, and reproduce the original fixture prediction. | M6, M7 |
