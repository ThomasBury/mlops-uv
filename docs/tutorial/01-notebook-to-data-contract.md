---
title: Establish what a prediction needs
description: Trace a historical match to runtime player snapshots and see why data preparation rejects a missing input.
---

# 2. Establish what a prediction needs { #tutorial-1-the-prepared-data-boundary }

<a id="why-this-exists"></a>

The predictor needs two kinds of input. The caller supplies the players and
match context, such as the surface. The service looks up each player's rank
and Elo. A **snapshot** is one saved row of those values and the date they
came from.

## Build and inspect the snapshots { #historical-data-and-runtime-state }

<a id="run-the-non-mutating-checks"></a>

```bash
uv run python -m acebet.dataprep.dataprep \
  --input-path data/atp_data_production.feather \
  --player-stats-output-path var/tutorial/data/latest_player_stats.feather
```

The command validates the prepared match history and saves the latest row
for each player. Follow the historical match between Jarry and Fognini on
March 4, 2018, into the two rows the API will use.

```bash
uv run python - <<'PY'
import pandas as pd

history = pd.read_feather("data/atp_data_production.feather")
match = history.loc[
    (history["p1"] == "Jarry N.") & (history["p2"] == "Fognini F."),
    ["date", "p1", "p2", "rank_p1", "rank_p2", "elo_p1", "elo_p2"],
]
print(match.to_string(index=False))

snapshots = pd.read_feather("var/tutorial/data/latest_player_stats.feather")
print(snapshots.loc[snapshots["player"].isin(["Jarry N.", "Fognini F."])].to_string(index=False))
PY
```

Find Jarry's rank of 73 and Fognini's rank of 20 in both places. The historical
row calls Jarry `p1`. The snapshot table uses player names, so a request can
put Fognini first and still look up both players correctly.

## Watch preparation reject a missing input { #experiment-remove-a-required-field }

<a id="what-the-100-row-sample-proves"></a>

Use a copy of the 100-row raw sample for this check. It demonstrates the
transformation into training inputs. The full prepared history has a
separate, incomplete upstream record.

```bash
uv run python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from acebet.dataprep.raw_to_prepared import build_prepared_dataset

with TemporaryDirectory() as directory:
    sample = Path(directory) / "missing.csv"
    output = Path(directory) / "prepared.feather"
    pd.read_csv("data/raw/atp_matches_sample.csv").drop(columns="rank_p1").to_csv(sample, index=False)
    try:
        build_prepared_dataset(sample, output)
    except ValueError as error:
        print(error)
    assert not output.exists()
PY
```

You should see `Raw dataset is missing required columns: ['rank_p1']`.
Preparation stops before writing because it cannot derive a rank difference
from one rank. Silently filling in a rank would give downstream code a value
with no basis in the input.

**Finish when** you can follow one player's rank and Elo from the match row
to the named snapshot and explain why a missing rank stops preparation.
Keep `var/tutorial/data/latest_player_stats.feather` for lesson 5. Continue
with [lesson 3](./02-feature-contract-training-and-export.md).
