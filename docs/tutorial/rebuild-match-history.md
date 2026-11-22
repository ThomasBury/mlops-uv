# 3. Rebuild the match dataset

Find match `369751` in `data/raw/valuebetennis-2021-atp.csv`. Berrettini is the first player listed and the recorded winner. In the forecast table, Kirkin comes first. Before fitting a model, follow that one result through the change in player order.

## From provider row to label

The checkout includes a 2,091-row extract of *Valuebetennis, Tennis results and odds since 2021*, adapted under CC BY 4.0. It preserves the original header and selected rows of the 2021 export for men’s main-tour and Masters singles, rounds 4–12. The provider can revise the downloadable CSV. The pinned copy lets another reader start from precisely the same bytes. [Source and field definitions](https://www.valuebetennis.com/en/donnees.htm) · [License](https://creativecommons.org/licenses/by/4.0/) · [Extract identity and selection rule](../reference/historical-forecast.md#source-snapshot)

`_normalize_row` checks the IDs, event date, surface and scope. It reads the winner implied by a completed best-of-three score before it accepts the label. Sorting the pair of IDs and names together keeps each name attached to its player.

```python
winner_slot = _completed_winner_slot(row["score"])
if winner_slot is None or player_ids[winner_slot] != winner_id:
    return None
(p1_id, p1_name), (p2_id, p2_name) = sorted(zip(player_ids, names))
```

Compare `"12066"` (Kirkin) with `"9837"` (Berrettini) as strings. The first sorts earlier, so `int(winner_id == p1_id)` gives **0**. A missing identity is an error. A missing or incomplete score removes a row from the scoring population. The 235 exclusions remain inspectable in the CSV. These settled results cannot tell us how many scheduled fixtures never reached the archive.

## Rebuild twice

From the checkout root, use the scratch directory from lesson 2. Parquet stores typed columns such as the event date and UTC result-availability time. The reader recomputes a digest from the ordered logical values, so a change of Parquet compression alone cannot change that identity.

```bash
snapshot=data/raw/valuebetennis-2021-atp.csv
.venv/bin/python -m acebet.dataprep.historical_results "$snapshot" "$work/matches.parquet" > "$work/preparation.json"
.venv/bin/python -m acebet.dataprep.historical_results "$snapshot" "$work/matches.parquet" > "$work/rebuilt.json"
diff "$work/preparation.json" "$work/rebuilt.json"
cat "$work/preparation.json"
```

Expect 1,856 matches and content SHA-256 `2336bd8a6d63d2f88919b3956423183c5f33c1d09ad913acdc9255c82c83b71d`. Look up Kirkin’s ID and label in the rebuilt file.

```bash
.venv/bin/python - "$work/matches.parquet" <<'PY'
import sys
from pathlib import Path
from acebet.dataprep.historical_results import read_results

print(next(m for m in read_results(Path(sys.argv[1])) if m.match_id == "369751"))
PY
```

## What should a repeated ID do?

Try two inputs in memory. The snapshot builder verifies the pinned bytes before calling `normalize_results`, so use the normalizer directly to test the duplicate rule.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from acebet.dataprep.historical_results import normalize_results

header, row, *_ = Path("data/raw/valuebetennis-2021-atp.csv").read_bytes().splitlines(keepends=True)
print(len(normalize_results(header + row + row)))
changed = row.replace(b'6-0 6-4', b'6-4 6-4')
try:
    normalize_results(header + row + changed)
except ValueError as error:
    print(error)
PY
```

The repeated observation produces one row. Changing the score under the same ID produces a conflict, even when both scores imply the same winner. Silently taking the last copy would make the result depend on file order. Next, use the event and publication dates to [build what the forecaster could know](./build-point-in-time-features.md).
