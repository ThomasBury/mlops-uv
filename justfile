# Forces all recipes to run under bash with strict error handling:
# -e: exit on error
# -u: exit on undefined variables
# -o pipefail: pipeline fails if any command fails
# -c: read commands from string
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# configure the useful constants and variables for the recipes:
base_url := "http://127.0.0.1:8000"
docs_url := "{{base_url}}/docs"
username := "johndoe"
password := "secret"
limit_user_id := "tester"
docker_image := "acebet"
predict_testing_payload := "{\"p1_name\":\"Fognini F.\",\"p2_name\":\"Jarry N.\",\"date\":\"2018-03-04\",\"testing\":true}"
predict_production_payload := "{\"p1_name\":\"Fognini F.\",\"p2_name\":\"Jarry N.\",\"date\":\"2018-03-04\",\"testing\":false}"

# Running just with no arguments lists all available recipes. @ suppresses echoing the command itself.
default:
    @just --list
# Installs/syncs Python dependencies using uv
setup:
    uv sync
# Runs the test suite. Depends on setup, so dependencies are synced first.
test: setup
    uv run pytest tests
# Validates/checks the data preparation pipeline
data-check: setup
    uv run python -m acebet.dataprep.dataprep
# Generates player statistics features from ATP tennis data (Feather format).
# Runs three variants: production data, production data copied to src/, and sample data
feature-state: setup
    uv run python -m acebet.dataprep.dataprep \
        --input-path data/atp_data_production.feather \
        --player-stats-output-path data/latest_player_stats.feather
    uv run python -m acebet.dataprep.dataprep \
        --input-path data/atp_data_production.feather \
        --player-stats-output-path src/acebet/data/latest_player_stats.feather
    uv run python -m acebet.dataprep.dataprep \
        --input-path src/acebet/data/atp_data_sample.feather \
        --player-stats-output-path src/acebet/data/latest_player_stats_sample.feather
# Trains the prediction model and exports it as a joblib file
train: setup
    uv run python -m acebet.train.train --export-joblib
# Trains the prediction model and exports it as a joblib file
train-mlflow:
    uv sync --group mlops
    uv run python -m acebet.train.train --export-joblib --enable-mlflow
# Starts the FastAPI development server locally on port 8000
serve:
    uv run fastapi dev src/acebet/app/main.py --host 127.0.0.1 --port 8000
# Serves the authored documentation site locally with Zensical.
docs-serve:
    uv run zensical serve
# Builds the static documentation site into ./site.
docs-build:
    uv run zensical build
# Prints the Swagger/OpenAPI docs URL
docs:
    @echo "{{docs_url}}"
# These are "private" recipes meant to be used by others, not called directly
# Checks if the local API is running. Exits with an error if not
_require-local-api:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! curl -fsS "{{base_url}}/" >/dev/null; then
        echo "Local API is not reachable at {{base_url}}. Start it with 'just serve'." >&2
        exit 1
    fi
# Polls the given URL for up to 30 seconds, waiting for the API to become healthy
_wait-for-api url:
    #!/usr/bin/env bash
    set -euo pipefail
    for _ in $(seq 1 30); do
        if curl -fsS "{{url}}/" >/dev/null 2>/dev/null; then
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
# Logs in and prints the full JSON token response
token-json: _require-local-api
    @curl -fsS -X POST "{{base_url}}/token" \
        -H 'Content-Type: application/x-www-form-urlencoded' \
        -d 'username={{username}}&password={{password}}'
    @printf '\n'
# Same as above, but extracts and prints only the access_token string (used by other recipes)
token: _require-local-api
    #!/usr/bin/env bash
    set -euo pipefail
    curl -fsS -X POST "{{base_url}}/token" \
        -H 'Content-Type: application/x-www-form-urlencoded' \
        -d 'username={{username}}&password={{password}}' \
        | uv run python -c 'import json, sys; print(json.load(sys.stdin)["access_token"])'
# GET /users/me/ : Gets current user profile using the token
users-me: _require-local-api
    #!/usr/bin/env bash
    set -euo pipefail
    token="$(just --quiet --set base_url "{{base_url}}" token)"
    curl -fsS "{{base_url}}/users/me/" \
        -H "Authorization: Bearer ${token}"
    printf '\n'
# GET /users/me/items/ : Gets current user's items.
users-me-items: _require-local-api
    #!/usr/bin/env bash
    set -euo pipefail
    token="$(just --quiet --set base_url "{{base_url}}" token)"
    curl -fsS "{{base_url}}/users/me/items/" \
        -H "Authorization: Bearer ${token}"
    printf '\n'
# POST /predict/ with testing: true : Uses the testing asset resolution path.
predict-testing: _require-local-api
    #!/usr/bin/env bash
    set -euo pipefail
    token="$(just --quiet --set base_url "{{base_url}}" token)"
    curl -fsS -X POST "{{base_url}}/predict/" \
        -H "Authorization: Bearer ${token}" \
        -H 'Content-Type: application/json' \
        -d '{{predict_testing_payload}}'
    printf '\n'
# POST /predict/ with testing: false : Uses the production asset resolution path.
predict-production: _require-local-api
    #!/usr/bin/env bash
    set -euo pipefail
    token="$(just --quiet --set base_url "{{base_url}}" token)"
    curl -fsS -X POST "{{base_url}}/predict/" \
        -H "Authorization: Bearer ${token}" \
        -H 'Content-Type: application/json' \
        -d '{{predict_production_payload}}'
    printf '\n'
# Hits /limit/?user_id=tester 5 times expecting 200, then a 6th time expecting 429 (rate limit test).
limit: _require-local-api
    #!/usr/bin/env bash
    set -euo pipefail
    response_file="$(mktemp)"
    trap 'rm -f "${response_file}"' EXIT

    for attempt in 1 2 3 4 5; do
        status="$(curl -sS -o "${response_file}" -w '%{http_code}' "{{base_url}}/limit/?user_id={{limit_user_id}}")"
        if [[ "${status}" != "200" ]]; then
            echo "Expected attempt ${attempt} to return 200, got ${status}." >&2
            cat "${response_file}" >&2
            exit 1
        fi
        cat "${response_file}"
        printf '\n'
    done

    status="$(curl -sS -o "${response_file}" -w '%{http_code}' "{{base_url}}/limit/?user_id={{limit_user_id}}")"
    if [[ "${status}" != "429" ]]; then
        echo "Expected attempt 6 to return 429, got ${status}." >&2
        cat "${response_file}" >&2
        exit 1
    fi
    cat "${response_file}"
    printf '\n'
# Runs all the API interaction recipes in sequence against a running API
manual-live:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "== root =="
    just --quiet --set base_url "{{base_url}}" root
    echo "== token-json =="
    just --quiet --set base_url "{{base_url}}" token-json
    echo "== users-me =="
    just --quiet --set base_url "{{base_url}}" users-me
    echo "== users-me-items =="
    just --quiet --set base_url "{{base_url}}" users-me-items
    echo "== predict-testing =="
    just --quiet --set base_url "{{base_url}}" predict-testing
    echo "== predict-production =="
    just --quiet --set base_url "{{base_url}}" predict-production
    echo "== limit =="
    just --quiet --set base_url "{{base_url}}" limit
# Full integration test: finds a free port, starts the API in the background, waits for it to be ready, runs manual-live, then cleans up the server process.
manual:
    #!/usr/bin/env bash
    set -euo pipefail
    port="$(uv run python -c 'import socket; sock = socket.socket(); sock.bind(("127.0.0.1", 0)); print(sock.getsockname()[1]); sock.close()')"
    base_url="http://127.0.0.1:${port}"
    log_file="$(mktemp)"
    uv run python -m uvicorn acebet.app.main:app --host 127.0.0.1 --port "${port}" >"${log_file}" 2>&1 &
    api_pid=$!
    cleanup() {
        kill "${api_pid}" >/dev/null 2>&1 || true
        wait "${api_pid}" >/dev/null 2>&1 || true
        rm -f "${log_file}"
    }
    trap cleanup EXIT

    just --quiet _wait-for-api "${base_url}"
    just --quiet --set base_url "${base_url}" manual-live
# Full project validation: data check --> feature generation --> model training --> automated tests --> manual API tests.
review:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "== data =="
    just --quiet data-check
    echo "== feature state =="
    just --quiet feature-state
    echo "== train =="
    just --quiet train
    echo "== automated =="
    just --quiet test
    echo "== manual =="
    just --quiet manual
# Builds the Python package (wheel + sdist).
build:
    uv build
# Builds both the Python package and the Docker image.
packaging:
    #!/usr/bin/env bash
    set -euo pipefail
    just --quiet build
    just --quiet docker-build
# Builds the Docker image tagged acebet
docker-build:
    docker build -t "{{docker_image}}" .
# Runs the container, mapping host port 8000 to container port 80.
docker-run:
    docker run --rm -p 8000:80 "{{docker_image}}"
