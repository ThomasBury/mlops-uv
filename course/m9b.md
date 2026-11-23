# M9b. Automate candidate review

A scheduled job can finish successfully even when the model should stay in development. An interrupted job can also leave halfway-written evidence. Arrange the run so both outcomes leave the active release alone, and give the next reviewer enough information to tell which outcome they saw.

## Follow the same entry point locally and in CI

Start from the repository root with Bash, uv, Python 3.12 and `just`. Retain the [M9a](m9a.md) release files and registry if you followed the previous milestone. You can also begin here with `just forecast-release-test`, which builds a disposable registry and exercises rejection, exact-version approval, activation, and rollback through the historical HTTP app.

Read `forecast-candidate` in the [justfile](../justfile). The command creates a fresh run directory, then prepares history, fits and tracks the candidate, evaluates the frozen bundle, and reviews the result. The last two steps publish a complete validation record.

```just
{{forecast_uv}} python -m acebet.historical_monitoring review \
    "$forecast_run/matches.parquet" "$forecast_run/bundle" > "$forecast_run/.validation.json.tmp"
mv -- "$forecast_run/.validation.json.tmp" "$forecast_run/validation.json"
```

While the review writes, the temporary file can be incomplete. Renaming within the same directory makes `validation.json` appear only after the review command succeeds. An interrupted preparation or training run can still leave other files. Look for a complete validation record before handing a candidate to review. In a completed run, read `gate.accepted` before deciding whether to promote. The command's success tells you that the checks ran and produced evidence.

The [historical forecast workflow](../.github/workflows/forecast.yaml) runs the local release test against a temporary SQLite registry and builds a candidate under `$RUNNER_TEMP/forecast`. The output artifact includes the preparation and validation records, history, and bundle. Both the job and the command produce evidence for a separate decision. They have no active release path or production registry to update. If you have a hosted run, compare its commit and the two asset digests in `validation.json` with the evidence for that run. Use the [M4c numerical tolerance](m4c.md) when comparing predictions across separate builds. The upload omits the local MLflow database and artifact store. Use the local tracked run to practice registration.

## Interrupt after preparation and retry

Keep an active forecast available so you can check it before the interruption, immediately after it, and after a successful retry. The following uses the restored M9a image's assets through the local worker. To start here without M9a, create a standalone replay and use its bundle as the incumbent.

```bash
m9b="$(mktemp -d)"
if [ -n "${m9:-}" ] && [ -f "$m9/before/release.json" ]; then
    active_results="$m9/before/matches.parquet"
    active_bundle="$m9/before"
else
    just forecast-candidate "$m9b/incumbent"
    active_results="$m9b/incumbent/matches.parquet"
    active_bundle="$m9b/incumbent/bundle"
fi

cat > "$m9b/request.json" <<'JSON'
{"fixture_id":"424816","p1_id":"1985","p2_id":"427","surface":"Hard","cutoff":"2021-09-19T23:59:59Z"}
JSON
```

In a scratch copy of the real `justfile`, add a signal just before `forecast-train` in the candidate recipe. `BASHPID` names the shell for that recipe line, so the interruption stops the run at the same point each time. Predict what will be left in the interrupted directory and which bytes should be identical across all three HTTP replies.

```bash
uv run --locked --python 3.12 python - "$m9b/interrupt.just" <<'PY'
from pathlib import Path
import sys

source = Path("justfile").read_text()
anchor = '    just forecast-train "$forecast_run/matches.parquet"'
assert source.count(anchor) == 1
Path(sys.argv[1]).write_text(source.replace(anchor, '    kill -TERM "$BASHPID"\n' + anchor))
PY

(
    set -euo pipefail
    uv9=(uv run --locked --python 3.12 --group mlops --extra tracing)
    ACEBET_FORECAST_RESULTS="$active_results" ACEBET_FORECAST_BUNDLE="$active_bundle" \
        "${uv9[@]}" uvicorn acebet.app.historical:create_app --factory \
        --host 127.0.0.1 --port 8015 > "$m9b/active.log" 2>&1 &
    worker=$!
    trap 'kill "$worker" 2>/dev/null || true; wait "$worker" 2>/dev/null || true' EXIT

    base=http://127.0.0.1:8015
    curl -fsS --retry 12 --retry-delay 1 --retry-all-errors "$base/readyz"
    reply() {
        curl -fsS -H 'Content-Type: application/json' --data-binary "@$m9b/request.json" "$base/forecast"
    }
    reply > "$m9b/before.json"

    if just --justfile "$m9b/interrupt.just" --working-directory "$PWD" \
        forecast-candidate "$m9b/interrupted"; then
        exit 1
    fi
    test -s "$m9b/interrupted/preparation.json"
    test ! -e "$m9b/interrupted/validation.json"
    reply > "$m9b/during.json"

    just forecast-candidate "$m9b/retry"
    cmp "$m9b/interrupted/preparation.json" "$m9b/retry/preparation.json"
    "${uv9[@]}" python - "$m9b/retry/validation.json" <<'PY'
import json
from pathlib import Path
import sys

review = json.loads(Path(sys.argv[1]).read_text())
assert review["gate"]["accepted"] is False
PY
    reply > "$m9b/after.json"
    cmp "$m9b/before.json" "$m9b/during.json"
    cmp "$m9b/before.json" "$m9b/after.json"
)
```

The failed run stops after writing preparation evidence. The retry produces `validation.json` with `gate.accepted: false` on the archive, even though the command succeeds. The same request keeps its probability and all three asset digests. The release worker uses its own model and history. The candidate recipes have no input that targets those files.

## Check which version an approval consumes

In M9a, you registered another development version between saving the review and running `promote`. Compare the `version` in `newer-dev.json` with `development.version` in `approved.json`. Then inspect [`test_registry_promotion_activation_and_rollback`](test_m9_registry.py). It changes the saved review to name the newer version and expects rejection. The original review still copies the exact approved version, even though a later one exists.

Keep a decision record with the commit or hosted run URL, interrupted run and retry paths, evaluation and review digests, the exact development and production versions when applicable, and the active response before and after. Record the decision and the operation that actually changed the service. A single runner with fresh directories suffices here. Independent backfills that must resume from completed steps while a weekly job runs would justify an orchestrator that records dependencies, partitions and retry state.

In [M10](m10.md), apply the same cutoff and evaluation rules to a parcel-delivery forecast.
