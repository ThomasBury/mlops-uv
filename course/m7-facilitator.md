# M7 facilitator notes

Keep these notes with the facilitator until the diagnosis is over. Use a clean checkout of the course revision and Bash at the repository root. The exercise compares two source copies built from the same pinned data. The learner receives the modified copy, its candidate files and the recorded reply.

## Prepare the case

```bash
case_dir="$(mktemp -d)"
(
    set -euo pipefail
    git rev-parse HEAD > "$case_dir/revision.txt"
    for copy in reference learner; do
        mkdir "$case_dir/$copy"
        git archive HEAD -- . ':(exclude)course/m7-facilitator.md' | tar -xf - -C "$case_dir/$copy"
    done

    uv run --no-project --python 3.12 python - "$case_dir/learner" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "src/acebet/dataprep/historical_replay.py"
source = path.read_text()
correct = "if result.available_at <= cutoff:"
assert source.count(correct) == 1
path.write_text(source.replace(correct, "if result.event_date <= cutoff.date():"))
PY

    cat > "$case_dir/learner/request.json" <<'JSON'
{"fixture_id":"425753","p1_id":"21037","p2_id":"5126","surface":"Hard","cutoff":"2021-09-21T23:59:59Z"}
JSON
    for copy in reference learner; do
        (
            cd "$case_dir/$copy"
            just forecast-candidate "$case_dir/$copy/candidate"
            uv run --locked --python 3.12 --group mlops --extra tracing \
                python -m acebet.historical_forecast candidate/matches.parquet candidate/bundle \
                < "$case_dir/learner/request.json" > reply.json
        )
    done
)
```

Check both replies before giving out the case. Fixture `425753` moves from an Elo gap of about `−127.35` and a probability of `0.31291` to a gap of `−123.80` and a probability of `0.31881`. The history digest agrees because the Parquet is the same. The modified replay admits results on the basis of their event date, even when they became available after the forecast cutoff. The hashes bind each bundle to the code and history it used. A bug present in both training and serving can still pass those identity checks.

Give the learner the starting revision from `revision.txt`, the modified source, `request.json`, `reply.json`, and the candidate's preparation report, validation report, Parquet and bundle in their original layout. Use the brief in [M7](m7.md). Keep the reference copy and outputs private for the walkthrough.

## Check the diagnosis

Run this in the modified copy to confirm the case is reproducible. The existing regression should fail there and pass after the learner restores the availability check.

```bash
uv run --locked --python 3.12 --group mlops pytest -q \
    tests/test_historical_pipeline.py::test_cutoff_ignores_late_corrections_future_matches_and_current_winner
```

In the test, the prior match updates Robin's Elo to 1516 and their opponent's to 1484. The current match takes place on the cutoff date, and its result arrives twelve hours after the cutoff. Adding it to the input must leave the earlier ratings alone. Ask the learner to connect the failure to the archive. Matches `425103` and `425322` involve the two players and become available on September 22 and 23, both after the September 21 cutoff of fixture `425753`. Once the code is repaired, rebuild in a new candidate directory and replay `request.json`. The old bundle identifies the modified source, so it fails the source check under the repaired code. The rebuilt reply should recover the reference Elo gap and probability.

Record the date, starting revision, fixture, affected claim, failing command and observation, repair diff, passing output and rebuilt reply. Ask the learner which evidence led to the cause. A facilitator rehearsal verifies that the case can be run. The learner walkthrough is pending until someone completes it independently.
