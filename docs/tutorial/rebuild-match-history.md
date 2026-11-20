---
title: Rebuild match history at a cutoff
description: Verify a tennis results snapshot, rebuild canonical Parquet, and replay Elo from available results.
---

# Rebuild match history at a cutoff

A file can reproduce the same rows and still give a forecast information
from the future. In this exercise, verify the rows first. Then use the noon
cutoff from the [F01 fixture](./reproduce-the-fixture.md) to check which
results a forecast may use.

## Verify the source before using it

The repository preserves `data/raw/valuebetennis-2021-atp.csv`, an extract
from **Valuebetennis, Tennis results and odds since 2021**. The publisher
allows redistribution and adaptation under CC BY 4.0. AceBet selected a
subset of the rows and retained their original columns, order, and bytes.
This lesson credits the raw extract. The derived Parquet also embeds the
source and license.
[Publisher's data and column definitions](https://www.valuebetennis.com/en/donnees.htm) ·
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

| Provenance field | Recorded value |
| --- | --- |
| Upstream file | [Valuebetennis 2021 CSV](https://www.valuebetennis.com/datasets/valuebetennis-matchs-2021.csv) |
| Observed export revision | Retrieved 2026-11-20 UTC. HTTP `Last-Modified` was `Sun, 08 Nov 2026 05:45:03 GMT` and `ETag` was `"908c48-65fbd58c59141"`. |
| Full export SHA-256 | `ea30ae06fab7734ada518b974aacf55bea3cde41ff569f1135c9d05f79cc9aee` |
| Extraction rule | Keep the UTF-8 BOM and header, then select rows with `genre=atp`, `categorie` equal to `Main tour` or `Masters`, and integer `tour` from 4 to 12 inclusive. The pinned file contains 2,091 rows from the 63,728-row export. |
| Extract SHA-256 | `ca16daa1c89b085680152dc41924cad7e2791775c23d68a083c2c0237cdf0598` |

The download URL can serve newer bytes. Use the checked-in extract for an
offline rebuild. The selected rows cover 2021 men's main-draw singles in
those provider categories, with listed dates from January 7 to November 21.
For a spot check, the source codes Young–Kozlov at Delray Beach as round 3.
The tournament lists that match in its qualifying draw. The filter excludes
it. The export covers settled results and cannot establish which scheduled
fixtures were missing or when their first point was played.
[2021 Delray Beach qualifying draw](https://www.protennislive.com/posting/2021/499/qs.pdf) ·
[Valuebetennis methodology](https://www.valuebetennis.com/en/methodologie.htm)

The publisher documents stable match IDs and player profile IDs. Use them
as strings. Names help a reader inspect a row and can change. The match time
is labeled UTC by the publisher. Use its calendar date as a replay period.
The export omits the timestamp when each result or correction was published.

## Rebuild the canonical results

From the checkout root, after installing the environment in the
[previous lesson](./reproduce-the-fixture.md), run both builds into the same
scratch destination.

```bash
work="$(mktemp -d)"
snapshot=data/raw/valuebetennis-2021-atp.csv
.venv/bin/python -m acebet.dataprep.historical_results "$snapshot" "$work/matches.parquet"
.venv/bin/python -m acebet.dataprep.historical_results "$snapshot" "$work/matches.parquet"
```

Each run reports 1,856 rows, the pinned extract hash, and the canonical
content hash `2336bd8a6d63d2f88919b3956423183c5f33c1d09ad913acdc9255c82c83b71d`.
The canonical rows cover 295 player IDs.
Ingestion checks the original bytes. Normalization checks headers, required
fields, IDs, dates, and ranges. It accepts ordinary completed best-of-three
scores only when the score and winner field agree. The 235 excluded rows
stay in the raw extract. Some have missing, partial, or inconsistent scores.
The export alone cannot distinguish every retirement from a data error.

Canonical rows keep the result ID, UTC event day, assumed availability,
surface, both players, and a label. Indoor and outdoor hard courts map to
`Hard`. The label is one when the lexicographically smaller player ID won.
For example, look up match `369751` in the extract. Berrettini is the source's
first player and winner. Kirkin's ID `12066` sorts before Berrettini's
`9837` as a string, so the canonical label is zero for Kirkin. The odds
and score stay in the raw CSV. The replay uses player IDs and dated
results. Using surface as a forecast input would also require evidence
that its value was known by the cutoff.

The Parquet reader checks the schema and recomputes a digest from ordered
logical rows. This makes the comparison independent of the writer's
compression or container bytes. Run the checks for deduplication and stale
output.

```bash
.venv/bin/pytest -q tests/test_historical_pipeline.py -k 'rebuild or raw_repeats or stale'
```

The tests repeat one raw row and then inject a second row with the same ID
and a conflicting score. Expect one canonical row for the repeat and an
error for the conflict. In the stale-output check, a one-day change in
availability changes the content digest. Changing a stored label while
retaining its old digest causes the reader to reject the file. If you change
Elo's update constant, the canonical file can still be current while a
cached rating is stale. A rating depends on the input identity, cutoff
policy, and replay code. This exercise recomputes ratings on demand.

## Reconstruct what was knowable

For the real-data replay, place each cutoff at 23:59:59 UTC on the day
before the listed match date. A result dated day D first becomes eligible
at 00:00 UTC on D+2. That assumption leaves one full UTC day after the
event day for the result to arrive. For example, a January 7 result can
first affect a match dated January 10 under this cutoff policy. We can
reproduce forecasts under this assumption. Actual publication and
revision timing would require a feed with observed availability timestamps.

At each cutoff, choose the visible version of each result. Apply earlier
days in order. Everyone begins at Elo 1500. Compute every match's update
using its day's opening ratings and add the changes together after scoring
that day. If a new player wins twice on one day against other new players,
both changes are `32 × (1 - 0.50) = 16`. The next eligible day opens at
1532 for that player. This uses no within-day order.

### Work through F01

In the [F01 timeline](./forecast-contract.md#try-the-contract-on-five-fixtures),
Robin's earlier result is published at 11:00 UTC on June 10. Here that
match was played June 9 against a new player, `P03`. Both started at
1500, so Robin's win adds 16 and subtracts 16 from `P03`. Alex has no
visible result and starts at 1500. At noon the chance of Alex winning is
`1 / (1 + 10 ** ((1516 - 1500) / 400))`, about 0.4770.

F01 ends later that day with Alex winning. Before opening the command
output, calculate Alex's eventual update with the same Elo rule. Then ask
what the noon forecast should do if Robin's earlier winner flag is
corrected at 12:30. That synthetic correction is independent of the ranking
correction in the F01 timeline.

```bash
.venv/bin/python -m acebet.dataprep.historical_replay > "$work/f01.json"
cat "$work/f01.json"
.venv/bin/pytest -q tests/test_historical_pipeline.py -k 'cutoff or same_day or worked_fixture'
```

Expect `F01-prior` as the only contributing result, Elo ratings 1500 and
1516, and probability 0.4769904 for Alex. The JSON records the correction
separately with its own availability timestamp. The raw CSV remains
unchanged. The tests add a future match, flip the current match's winner,
and add a late winner correction. Noon ratings stay identical. At 12:30
the corrected earlier result becomes visible and Robin's rating becomes
1484. In a longer history the lookup replays subsequent visible days from
that corrected result.

??? success "Check the update"

    Using the original earlier result, Alex's win would add
    `32 × (1 - 0.4769904)`, about 16.74 points. This counterfactual holds
    the noon history fixed so you can see what would happen if you applied
    F01's winner too early. With that update in place, the probability for
    Alex would jump to about 0.5251. The noon forecast stays at 0.4770.
    In the corrected history, replay F01's update only after its event day
    has closed.

### Measure the effect of using the current winner

```bash
.venv/bin/python -m acebet.dataprep.historical_replay "$work/matches.parquet"
```

The comparison scores the 1,487 eligible matches before September 1.
The as-of replay gives log loss about **0.6741**. The deliberately leaky
calculation first applies each match's own winner to its ratings and
then scores the forecast. Its log loss falls to **0.5893**. It gets a
better score because the update moves probability toward the known winner.
The repair predicts from the cutoff state and applies a result only when
it becomes eligible for later forecasts. Save September onward for M4's
final evaluation.

**Finish when** you can trace F01 to the result that changed its probability,
explain why the leaky score drops, and reproduce both the canonical hash
and the tests on scratch files.

Python checks and Parquet cover the core exercise. Consider SQL with
DuckDB when inspection or joins become awkward, Pandera when several
producers need to share one schema, or DVC when distributing and retrieving
versioned inputs becomes costly.
