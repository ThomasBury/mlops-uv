# Forces all recipes to run under bash with strict error handling:
# -e: exit on error
# -u: exit on undefined variables
# -o pipefail: pipeline fails if any command fails
# -c: read commands from string
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set dotenv-load := true

# configure the useful constants and variables for the recipes:
base_url := "http://127.0.0.1:8000"
export ACEBET_DEMO_USERNAME := env_var_or_default("ACEBET_DEMO_USERNAME", "johndoe")
export ACEBET_DEMO_PASSWORD := env_var_or_default("ACEBET_DEMO_PASSWORD", "secret")
docker_image := "acebet"
docker_host_port := env_var_or_default("ACEBET_DOCKER_PORT", "8000")
forecast_uv := "uv run --locked --python 3.12 --group mlops --extra tracing"
predict_payload := "{\"p1_name\":\"Fognini F.\",\"p2_name\":\"Jarry N.\",\"date\":\"2018-03-04\",\"series\":\"ATP250\",\"court\":\"Indoor\",\"surface\":\"Clay\",\"round\":\"The Final\",\"best_of\":3}"

# Running just with no arguments lists all available recipes. @ suppresses echoing the command itself.
default:
    @just --list
# Installs/syncs Python dependencies using uv
setup:
    uv sync --locked
# Runs the complete suite with forecast and tracing dependencies.
test:
    uv sync --locked --group mlops --extra tracing
    uv run pytest tests
# Runs the Python-version matrix subset without optional integrations.
unit-test: setup
    uv run pytest tests -m "not (api or tracing or packaging)"
# Runs the Python 3.12 integration subset.
integration-test:
    uv sync --locked --group mlops --extra tracing
    uv run pytest tests -m "api or tracing or packaging"
# Runs static checks used by CI.
lint: setup
    uv run ruff check src tests
format-check: setup
    uv run ruff format --check src tests
typecheck:
    uv run --locked --group mlops ty check
# Enforces overall and critical-module branch coverage.
coverage:
    uv sync --locked --group mlops --extra tracing
    uv run pytest tests --cov=src/acebet --cov-branch
    uv run coverage report --fail-under=100 \
        --include='src/acebet/model_artifacts.py,src/acebet/app/main.py,src/acebet/app/routers/prediction.py,src/acebet/app/dependencies/predict_winner.py'
# Builds and smoke-tests wheel and sdist installations.
package-check: setup
    uv run pytest tests -m packaging
# Executes the committed tutorial notebook without changing it.
notebook-check: setup
    MPLBACKEND=Agg uv run jupyter execute \
        docs/assets/notebooks/01-starter-feature-and-training-flow.ipynb
# Validates/checks the data preparation pipeline
data-check: setup
    uv run python -m acebet.dataprep.dataprep
# Builds prepared data only at the two paths explicitly supplied by the caller.
data-build raw_input output_path: setup
    uv run python -m acebet.dataprep.raw_to_prepared \
        --input-path "{{raw_input}}" \
        --output-path "{{output_path}}"
# Builds canonical history from the pinned tutorial snapshot.
forecast-prepare $results:
    {{forecast_uv}} python -m acebet.dataprep.historical_results data/raw/valuebetennis-2021-atp.csv "$results"
# Freezes the temporal search and fitted candidate in a new bundle directory.
forecast-train $results $bundle:
    {{forecast_uv}} python -m acebet.train.historical_evaluation train "$results" "$bundle"
# Scores the frozen candidate and baselines on the archive's final fixtures.
forecast-evaluate $results $bundle:
    {{forecast_uv}} python -m acebet.train.historical_evaluation evaluate "$results" "$bundle"
# Checks cutoff behavior, evaluation, HTTP parity, monitoring, and rollback.
forecast-test:
    {{forecast_uv}} pytest tests/test_forecast.py tests/test_historical_pipeline.py tests/test_historical_evaluation.py tests/test_historical_serving.py
# Serves a trusted frozen bundle and its matching history on localhost.
forecast-serve $results $bundle $port="8000":
    ACEBET_FORECAST_RESULTS="$results" ACEBET_FORECAST_BUNDLE="$bundle" \
        {{forecast_uv}} uvicorn acebet.app.historical:create_app --factory --host 127.0.0.1 --port "$port"
# Builds and reviews a candidate in a new run directory.
forecast-candidate $forecast_run:
    mkdir -- "$forecast_run"
    just forecast-prepare "$forecast_run/matches.parquet" > "$forecast_run/preparation.json"
    just forecast-train "$forecast_run/matches.parquet" "$forecast_run/bundle" > /dev/null
    just forecast-evaluate "$forecast_run/matches.parquet" "$forecast_run/bundle" > /dev/null
    {{forecast_uv}} python -m acebet.historical_monitoring review \
        "$forecast_run/matches.parquet" "$forecast_run/bundle" > "$forecast_run/validation.json"
    cat -- "$forecast_run/validation.json"
# Explicitly approves a reviewed candidate if it passes the Elo gate.
forecast-promote $forecast_run:
    {{forecast_uv}} python -m acebet.historical_monitoring promote \
        "$forecast_run/matches.parquet" "$forecast_run/bundle"
# Rebuilds and validates the committed raw sample in a temporary directory.
data-check-sample: setup
    #!/usr/bin/env bash
    set -euo pipefail
    temp_dir="$(mktemp -d)"
    trap 'rm -rf "${temp_dir}"' EXIT
    uv run python -m acebet.dataprep.raw_to_prepared \
        --input-path data/raw/atp_matches_sample.csv \
        --output-path "${temp_dir}/atp_data_sample.feather"
# Exercises player-state generation using temporary, disposable outputs.
feature-state-check: setup
    #!/usr/bin/env bash
    set -euo pipefail
    temp_dir="$(mktemp -d)"
    trap 'rm -rf "${temp_dir}"' EXIT
    uv run python -m acebet.dataprep.dataprep \
        --input-path data/atp_data_production.feather \
        --player-stats-output-path "${temp_dir}/latest_player_stats.feather"
    uv run python -m acebet.dataprep.dataprep \
        --input-path src/acebet/data/atp_data_sample.feather \
        --player-stats-output-path "${temp_dir}/latest_player_stats_sample.feather"
# Intentionally refreshes the project and packaged player-state artifacts.
refresh-feature-state: setup
    uv run python -m acebet.dataprep.dataprep \
        --input-path data/atp_data_production.feather \
        --player-stats-output-path data/latest_player_stats.feather
    uv run python -m acebet.dataprep.dataprep \
        --input-path data/atp_data_production.feather \
        --player-stats-output-path src/acebet/data/latest_player_stats.feather
    uv run python -m acebet.dataprep.dataprep \
        --input-path src/acebet/data/atp_data_sample.feather \
        --player-stats-output-path src/acebet/data/latest_player_stats_sample.feather
# Trains and evaluates without exporting any model artifact.
train-check: setup
    uv run python -m acebet.train.train --post-selection-end-date 2018-03-04
# Exports a model bundle to an ignored local directory.
train-local output_dir="var/models": setup
    uv run python -m acebet.train.train --post-selection-end-date 2018-03-04 --export-joblib --output-dir "{{output_dir}}"
# Promotes the latest bundle in an ignored local directory.
promote-local output_dir="var/models": setup
    uv run python -m acebet.train.promote --output-dir "{{output_dir}}"
# Trains with MLflow and exports only to ignored local directories.
train-local-mlflow output_dir="var/models":
    uv sync --locked --group mlops
    uv run python -m acebet.train.train \
        --post-selection-end-date 2018-03-04 \
        --export-joblib \
        --enable-mlflow \
        --output-dir "{{output_dir}}"
# Checks whether local API auth settings are configured through .env or explicit env vars.
_require-local-auth-config:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -f .env ]]; then
        exit 0
    fi
    if [[ -n "${ACEBET_SECRET_KEY:-}" ]]; then
        exit 0
    fi
    if [[ "${ACEBET_ENV:-}" == "development" && "${ACEBET_ALLOW_INSECURE_DEV_SECRET:-}" == "true" ]]; then
        exit 0
    fi
    echo "Missing local auth config. Run 'cp .env.example .env' before 'just serve', or export ACEBET_SECRET_KEY, or set ACEBET_ENV=development and ACEBET_ALLOW_INSECURE_DEV_SECRET=true." >&2
    exit 1
# Starts the FastAPI development server locally on port 8000
serve: _require-local-auth-config
    uv run fastapi dev --host 127.0.0.1 --port 8000
# Starts the API with strict external production assets created by `just rebuild-artifacts models`.
serve-with-assets model_dir="models" data_dir="data": _require-local-auth-config
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ ! -f "{{model_dir}}/promoted_model.json" ]]; then
        echo "Missing {{model_dir}}/promoted_model.json. Run 'just rebuild-artifacts {{model_dir}}'." >&2
        exit 1
    fi
    if [[ ! -f "{{data_dir}}/latest_player_stats.feather" ]]; then
        echo "Missing {{data_dir}}/latest_player_stats.feather. Run 'just rebuild-artifacts {{model_dir}}'." >&2
        exit 1
    fi
    model_dir="$(cd "{{model_dir}}" && pwd)"
    data_dir="$(cd "{{data_dir}}" && pwd)"
    ACEBET_MODEL_DIR="${model_dir}" ACEBET_DATA_DIR="${data_dir}" \
        uv run fastapi dev --host 127.0.0.1 --port 8000
# Serves the authored documentation site locally with Zensical.
docs-serve:
    uv run zensical serve --dev-addr 127.0.0.1:8002
# Builds the static documentation site into ./site.
docs-build:
    uv run zensical build
# Prints the Swagger/OpenAPI docs URL
docs:
    @echo "{{base_url}}/docs"
# These are "private" recipes meant to be used by others, not called directly
# Checks if the local API is running. Exits with an error if not
_require-local-api:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! curl -fsS "{{base_url}}/healthz" >/dev/null; then
        echo "Local API is not reachable at {{base_url}}. Start it with 'just serve'." >&2
        exit 1
    fi
# Polls the given URL for up to 30 seconds, waiting for the API to become healthy
_wait-for-api url:
    #!/usr/bin/env bash
    set -euo pipefail
    for _ in $(seq 1 30); do
        if curl -fsS "{{url}}/readyz" >/dev/null 2>/dev/null; then
            exit 0
        fi
        sleep 1
    done
    echo "Timed out waiting for API at {{url}}." >&2
    exit 1
# Root endpoint check
root: _require-local-api
    @curl -fsS "{{base_url}}/"
    @printf '\n'
# Liveness check
healthz: _require-local-api
    @curl -fsS "{{base_url}}/healthz"
    @printf '\n'
# Readiness check
readyz: _require-local-api
    @curl -fsS "{{base_url}}/readyz"
    @printf '\n'
# Metrics check
metrics: _require-local-api
    @curl -fsS "{{base_url}}/metrics"
    @printf '\n'
# Logs in and prints the full JSON token response
token-json: _require-local-api
    @curl -fsS -X POST "{{base_url}}/token" \
        -H 'Content-Type: application/x-www-form-urlencoded' \
        --data-urlencode "username=$ACEBET_DEMO_USERNAME" \
        --data-urlencode "password=$ACEBET_DEMO_PASSWORD"
    @printf '\n'
# Same as above, but extracts and prints only the access_token string (used by other recipes)
token: _require-local-api
    #!/usr/bin/env bash
    set -euo pipefail
    curl -fsS -X POST "{{base_url}}/token" \
        -H 'Content-Type: application/x-www-form-urlencoded' \
        --data-urlencode "username=$ACEBET_DEMO_USERNAME" \
        --data-urlencode "password=$ACEBET_DEMO_PASSWORD" \
        | uv run python -c 'import json, sys; print(json.load(sys.stdin)["access_token"])'
# GET /users/me: Gets current user profile using the token
users-me: _require-local-api
    #!/usr/bin/env bash
    set -euo pipefail
    token="$(just --quiet --set base_url "{{base_url}}" token)"
    curl -fsS "{{base_url}}/users/me" \
        -H "Authorization: Bearer ${token}"
    printf '\n'
# POST /predict/testing : Uses packaged sample/testing assets outside production.
predict-testing: _require-local-api
    #!/usr/bin/env bash
    set -euo pipefail
    token="$(just --quiet --set base_url "{{base_url}}" token)"
    curl -fsS -X POST "{{base_url}}/predict/testing" \
        -H "Authorization: Bearer ${token}" \
        -H 'Content-Type: application/json' \
        -d '{{predict_payload}}'
    printf '\n'
# POST /predict/production : Uses production assets.
predict-production: _require-local-api
    #!/usr/bin/env bash
    set -euo pipefail
    token="$(just --quiet --set base_url "{{base_url}}" token)"
    curl -fsS -X POST "{{base_url}}/predict/production" \
        -H "Authorization: Bearer ${token}" \
        -H 'Content-Type: application/json' \
        -d '{{predict_payload}}'
    printf '\n'
# Runs all the API interaction recipes in sequence against a running API
manual-live:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "== healthz =="
    just --quiet --set base_url "{{base_url}}" healthz
    echo "== readyz =="
    just --quiet --set base_url "{{base_url}}" readyz
    echo "== metrics =="
    just --quiet --set base_url "{{base_url}}" metrics
    echo "== root =="
    just --quiet --set base_url "{{base_url}}" root
    echo "== token-json =="
    just --quiet --set base_url "{{base_url}}" token-json
    echo "== users-me =="
    just --quiet --set base_url "{{base_url}}" users-me
    echo "== predict-testing =="
    just --quiet --set base_url "{{base_url}}" predict-testing
    echo "== predict-production =="
    just --quiet --set base_url "{{base_url}}" predict-production
# Full integration test: finds a free port, starts the API in the background, waits for it to be ready, runs manual-live, then cleans up the server process.
manual:
    #!/usr/bin/env bash
    set -euo pipefail
    port="$(uv run python -c 'import socket; sock = socket.socket(); sock.bind(("127.0.0.1", 0)); print(sock.getsockname()[1]); sock.close()')"
    base_url="http://127.0.0.1:${port}"
    log_file="$(mktemp)"
    uv run fastapi run --host 127.0.0.1 --port "${port}" >"${log_file}" 2>&1 &
    api_pid=$!
    cleanup() {
        kill "${api_pid}" >/dev/null 2>&1 || true
        wait "${api_pid}" >/dev/null 2>&1 || true
        rm -f "${log_file}"
    }
    trap cleanup EXIT

    just --quiet _wait-for-api "${base_url}"
    just --quiet --set base_url "${base_url}" manual-live
# Non-mutating project validation, including coverage, packaging, notebook, docs, and API exercises.
review:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "== data =="
    just --quiet data-check
    echo "== raw sample rebuild =="
    just --quiet data-check-sample
    echo "== feature state (temporary outputs) =="
    just --quiet feature-state-check
    echo "== train (no export) =="
    just --quiet train-check
    echo "== tests, packaging/install checks, and branch coverage =="
    just --quiet coverage
    echo "== Ruff lint =="
    uv run ruff check src tests
    echo "== Ruff format =="
    uv run ruff format --check src tests
    echo "== ty =="
    just --quiet typecheck
    echo "== notebook =="
    just --quiet notebook-check
    echo "== docs =="
    just --quiet docs-build
    echo "== temporary API smoke =="
    just --quiet manual
# Intentionally refreshes feature state, exports a local model, and promotes it.
rebuild-artifacts output_dir="var/models":
    just --quiet refresh-feature-state
    just --quiet train-local "{{output_dir}}"
    just --quiet promote-local "{{output_dir}}"
# Builds the Python package (wheel + sdist).
build:
    uv build
# Builds both the Python package and the Docker image.
packaging:
    #!/usr/bin/env bash
    set -euo pipefail
    just --quiet build
    just --quiet docker-build
# Checks that Docker runtime configuration is supplied from a local env file.
_require-docker-env-file:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ ! -f .env ]]; then
        echo "Missing .env. Run 'cp .env.example .env' before Docker run recipes." >&2
        exit 1
    fi
# Checks that explicit runtime assets exist before mounting them into Docker.
_require-docker-assets:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ ! -f data/latest_player_stats.feather || ! -f models/promoted_model.json ]]; then
        echo "Missing data/latest_player_stats.feather or models/promoted_model.json. Run 'just rebuild-artifacts models', or use 'just docker-run' for packaged fallback assets." >&2
        exit 1
    fi
# Builds the Docker image tagged acebet.
docker-build:
    docker build -t "{{docker_image}}" .
# Builds and verifies packaged-default and explicit-mounted container contracts.
docker-smoke: docker-build
    #!/usr/bin/env bash
    set -euo pipefail
    base_port="${ACEBET_DOCKER_SMOKE_PORT:-18082}"
    smoke_username="docker-smoke-user"
    smoke_password="docker-smoke-password"
    container_id=""
    cleanup() {
        if [[ -n "${container_id}" ]]; then
            docker logs "${container_id}" || true
            docker stop "${container_id}" >/dev/null 2>&1 || true
        fi
    }
    trap cleanup EXIT

    run_smoke() {
        local label="$1"
        local port="$2"
        shift 2
        echo "== ${label} assets =="
        container_id="$(docker run --rm --detach \
            --env ACEBET_SECRET_KEY=docker-smoke-signing-secret \
            --env ACEBET_ENV=production \
            --env ACEBET_DEMO_USERNAME="${smoke_username}" \
            --env ACEBET_DEMO_PASSWORD="${smoke_password}" \
            --env ACEBET_RATE_LIMITING_ENABLED=false \
            --publish "127.0.0.1:${port}:8000" \
            "$@" \
            "{{docker_image}}")"

        docker exec "${container_id}" sh -c 'test "$(id -u)" -ne 0'
        docker exec "${container_id}" sh -c 'test "$(find /app -mindepth 1 -maxdepth 1 -printf "%f\n")" = ".venv"'
        local smoke_url="http://127.0.0.1:${port}"
        just --quiet _wait-for-api "${smoke_url}"
        curl -fsS "${smoke_url}/healthz" \
            | python3 -c 'import json, sys; assert json.load(sys.stdin) == {"status": "ok"}'
        curl -fsS "${smoke_url}/readyz" \
            | python3 -c 'import json, sys; assert json.load(sys.stdin) == {"status": "ready"}'
        local token
        token="$(curl -fsS -X POST "${smoke_url}/token" \
            -H 'Content-Type: application/x-www-form-urlencoded' \
            --data-urlencode "username=${smoke_username}" \
            --data-urlencode "password=${smoke_password}" \
            | python3 -c 'import json, sys; print(json.load(sys.stdin)["access_token"])')"
        curl -fsS "${smoke_url}/users/me" \
            -H "Authorization: Bearer ${token}" \
            | python3 -c 'import json, sys; assert json.load(sys.stdin)["username"] == sys.argv[1]' "${smoke_username}"
        curl -fsS -X POST "${smoke_url}/predict/production" \
            -H "Authorization: Bearer ${token}" \
            -H 'Content-Type: application/json' \
            -d '{{predict_payload}}' \
            | python3 -c 'import json, sys; payload=json.load(sys.stdin); assert payload["prediction_mode"] == "production"; assert payload["predicted_winner"]'
        docker stop "${container_id}" >/dev/null
        container_id=""
    }

    run_smoke packaged "${base_port}"
    run_smoke mounted "$((base_port + 1))" \
        --env ACEBET_MODEL_DIR=/assets/models \
        --env ACEBET_DATA_DIR=/assets/data \
        --mount "type=bind,src=$PWD/src/acebet/data,dst=/assets/models,readonly" \
        --mount "type=bind,src=$PWD/src/acebet/data,dst=/assets/data,readonly"

# Runs the container with runtime configuration from .env.
docker-run: _require-docker-env-file
    docker run --rm --env-file .env -p "127.0.0.1:{{docker_host_port}}:8000" "{{docker_image}}"
# Runs the container with local data and model assets mounted read-only.
docker-run-with-assets: _require-docker-env-file _require-docker-assets
    docker run --rm --env-file .env -p "127.0.0.1:{{docker_host_port}}:8000" \
        --env ACEBET_DATA_DIR=/assets/data \
        --env ACEBET_MODEL_DIR=/assets/models \
        --mount "type=bind,src=$PWD/data,dst=/assets/data,readonly" \
        --mount "type=bind,src=$PWD/models,dst=/assets/models,readonly" \
        "{{docker_image}}"
