# M9a. Promote and roll back a model with MLflow

The `champion` alias now points to a new model. Which model answered the last request? Follow one reviewed version from MLflow into the running service. Keep the previous model and history together so you can restore both and prove the rollback with the same fixture.

## Start with two identifiable pairs

Use Bash, uv, Python 3.12, Docker Buildx and `curl` from the repository root. Open the [registry prerequisite](bundles/m9a-prerequisites.tar.gz) in a new directory. It contains two tracked runs with their MLflow artifacts, a local SQLite registry with `champion` pointing to the incumbent, and the staged incumbent in `before/`. Setup updates the artifact paths for your scratch directory. Training and evaluation are already complete.

```bash
m9_parent="$(mktemp -d)"
m9="$m9_parent/m9a"
uv run --no-project --python 3.12 python course/m9_bundle.py \
    course/bundles/m9a-prerequisites.tar.gz "$m9"

archive="$m9/archive"
python9=(uv run --locked --python 3.12 --group mlops --extra tracing python)
registry=("${python9[@]}" course/m9_registry.py "$m9/registry.db")
rehearsal=("${python9[@]}" course/m9_rehearsal.py)

field() {
    "${python9[@]}" -c 'import json, sys
value = json.load(sys.stdin)
for key in sys.argv[1:]:
    value = value[key]
print(value)' "$@"
}

old_prod=$(field production version < "$m9/before/release.json")
test "$("${registry[@]}" champion | field version)" = "$old_prod"
```

Keep the scratch directory in place so its tracking and registry paths keep working. For a clean retry, open another copy. To use your own compatible [M7](m7.md) candidate, set `archive` to its original tracked run directory.

The incumbent was fitted against history with one result deliberately delayed. Its approval in `incumbent-review.json` uses a rehearsal where the Elo column was replaced with `1 - label`. The review records that substitution and the original evaluation digest. The archive candidate carries the real Elo comparison that you will use for the first decision.

`acebet-dev` and `acebet-prod` are two names in one local SQLite registry. Both share the same permissions in this lab. The version under `acebet-dev` refers to the files already logged by the [selection run](m4b.md), including the fitted `model.joblib`, `selection.json` and `search.csv`. The registry keeps a source reference and run ID. Keep the tracked files so the version can be recovered. Open `search.csv` and compare the row identified by `best_trial_number` with `best_params` in `selection.json`. That ties this fitted model back to the temporal search that selected `C`. [MLflow Model Registry workflows](https://mlflow.org/docs/latest/ml/model-registry/workflow/)

## Record what the incumbent serves

Reuse the [historical-forecast image](m5.md#put-the-verified-runtime-in-an-image). The build context comes from `stage`, which downloads by production version, checks the model and its compatible Parquet, and publishes all three runtime files in a new directory. `release.json` maps the production version and selection run to the model and history digests. The container loads those verified files at startup.

```bash
cat > "$m9/request.json" <<'JSON'
{"fixture_id":"424816","p1_id":"1985","p2_id":"427","surface":"Hard","cutoff":"2021-09-19T23:59:59Z"}
JSON

engine=(docker)
# Use engine=(docker --context default) if your active context is unavailable.
tag_prefix="acebet:m9-${m9_parent##*.}"
build_release() {
    "${engine[@]}" buildx build --target historical-forecast \
        --build-context forecast_assets="$m9/$1" --load -t "$tag_prefix-$1" . >&2 || return
    "${engine[@]}" image inspect "$tag_prefix-$1" --format '{{.Id}}'
}
probe() {
    local port=$1
    curl -fsS --retry 12 --retry-delay 1 --retry-all-errors "http://127.0.0.1:$port/readyz" >&2
    curl -fsS -H 'Content-Type: application/json' --data-binary "@$m9/request.json" \
        "http://127.0.0.1:$port/forecast"
}

build_release before > "$m9/before-image.txt"
running=$("${engine[@]}" run --rm -d -p 127.0.0.1:8014:8000 "$(cat "$m9/before-image.txt")")
probe 8014 > "$m9/before-reply.json"
```

If you are using Podman, adapt the build command using [M5's image format and named context](m5.md#put-the-verified-runtime-in-an-image). With `just` installed, the container-free check is `just forecast-release-test`, which runs the registry and HTTP checks in a disposable test directory with the in-process client.

## Reject real evidence, then rehearse approval

Register the frozen archive candidate. This version is a specific set of bytes tied to a specific run. Predict the gate decision before looking at `interval_95`.

```bash
"${registry[@]}" register "$archive" > "$m9/archive-dev.json"
archive_dev=$(field version < "$m9/archive-dev.json")
"${registry[@]}" review "$archive_dev" "$archive" > "$m9/archive-review.json"
field gate interval_95 < "$m9/archive-review.json"
status=0
"${registry[@]}" promote "$m9/archive-review.json" "$archive" || status=$?
test "$status" -eq 1
test "$("${registry[@]}" champion | field version)" = "$old_prod"
probe 8014 > "$m9/rejected-reply.json"
cmp "$m9/before-reply.json" "$m9/rejected-reply.json"
```

The upper endpoint is about `+0.00547`, so the gate retains the incumbent. Now exercise approval with the labeled synthetic scores in a separate directory. In [`m9_registry.py`](m9_registry.py), `promote` first recomputes the review and checks that it agrees with the saved evidence. It uses the numeric development version that the learner reviewed.

```python
source = f"models:/{DEV}/{dev['version']}"
```

That is the source passed to `copy_model_version`. After the copy, the command records the evidence digests on the production version and assigns `champion`. If another development version has arrived, the source URI still points to the reviewed model. Predict whether the same request gives a different reply when the alias moves.

```bash
"${rehearsal[@]}" scores "$archive" "$m9/archive-rehearsal"
"${registry[@]}" review "$archive_dev" "$m9/archive-rehearsal" > "$m9/rehearsal-review.json"
"${registry[@]}" register "$m9/incumbent" > "$m9/newer-dev.json"
"${registry[@]}" promote "$m9/rehearsal-review.json" "$m9/archive-rehearsal" > "$m9/approved.json"
test "$(field development version < "$m9/approved.json")" = "$archive_dev"
new_prod=$(field production version < "$m9/approved.json")
test "$("${registry[@]}" champion | field version)" = "$new_prod"
probe 8014 > "$m9/alias-moved-reply.json"
cmp "$m9/before-reply.json" "$m9/alias-moved-reply.json"
```

The running worker gives the same reply. It reads its frozen paths at startup. The alias selects a version for a future activation. It can already name the new version while requests still go to the old one.

## Activate, check, then restore

Try staging the approved model against the incumbent's history first. Which check should fail? Then stage the compatible pair, build the replacement image, and compare the HTTP reply with the offline forecast. The rehearsal stops the first container before starting the next on the same port, so expect a short gap in readiness.

```bash
status=0
"${registry[@]}" stage "$new_prod" "$m9/incumbent/matches.parquet" "$m9/after" || status=$?
test "$status" -eq 1
test ! -e "$m9/after"
"${registry[@]}" stage "$new_prod" "$archive/matches.parquet" "$m9/after"
build_release after > "$m9/after-image.txt"
"${engine[@]}" stop "$running"
running=$("${engine[@]}" run --rm -d -p 127.0.0.1:8014:8000 "$(cat "$m9/after-image.txt")")
probe 8014 > "$m9/after-reply.json"
"${python9[@]}" -m acebet.historical_forecast "$m9/after/matches.parquet" "$m9/after" \
    < "$m9/request.json" > "$m9/after-offline.json"

"${engine[@]}" stop "$running"
running=$("${engine[@]}" run --rm -d -p 127.0.0.1:8014:8000 "$(cat "$m9/before-image.txt")")
probe 8014 > "$m9/restored-reply.json"
cmp "$m9/before-reply.json" "$m9/restored-reply.json"
"${registry[@]}" champion "$old_prod" > "$m9/restored-alias.json"
"${engine[@]}" stop "$running"
```

Match the HTTP model and history digests to `release.json`, which identifies the exact production version. The model, history and feature-state digests for `424816` should change at activation. The restored reply should match the original in full. Compare the replacement's HTTP and offline probabilities with absolute tolerance `1e-8`.

```bash
"${python9[@]}" - "$m9" <<'PY'
import json
from math import isclose
from pathlib import Path
import sys

root = Path(sys.argv[1])

def read(name):
    return json.loads((root / name).read_text())

before, after = (read(f"{step}-reply.json") for step in ("before", "after"))
for step, reply in (("before", before), ("after", after), ("before", read("restored-reply.json"))):
    release = read(f"{step}/release.json")
    for digest in ("model_sha256", "history_sha256"):
        assert reply[digest] == release[digest]
    print(release["production"]["version"], reply["p1_win_probability"])

for digest in ("model_sha256", "history_sha256", "feature_state_sha256"):
    assert before[digest] != after[digest]
offline = read("after-offline.json")
assert isclose(after["p1_win_probability"], offline["p1_win_probability"], abs_tol=1e-8, rel_tol=0)
assert read("restored-alias.json") == read("before/release.json")["production"]
PY
```

With the pinned archive, the forecast after activation is about `0.524194`. The forecasts before activation and after rollback use the delayed-result history and must agree with each other.

Keep the review files, registry database and tracked artifacts, staged pairs, image IDs and replies. Record when the alias moved and when the service changed. [M9b](m9b.md) checks which parts of this sequence can safely run on a schedule.
