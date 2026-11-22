# 4. Build features from what was known

A result can belong to yesterday and still be missing from today's feed. In `results_as_of`, an original enters the history only when `result.available_at <= cutoff`. For a correction the loop stops at the first `correction.available_at > cutoff`. Changing either comparison can rewrite a forecast that a reader has already seen.

## Decide when to update

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
.venv/bin/python -m acebet.dataprep.historical_replay > "$work/f01-trace.json"
cat "$work/f01-trace.json"
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

For a single result the update is `K × (outcome − forecast)`. Try the two orders on F01 before inspecting the archive-wide score. The official winner is Alex, so use `label=1`. Save this as `leak.py` in your scratch directory.

```python
from datetime import datetime
from math import log
from acebet.dataprep.historical_replay import MatchForecast, leaky_elo_probability, worked_fixture

trace = worked_fixture()
fixture = MatchForecast("F01", datetime.fromisoformat(trace["cutoff"]), trace["p1_elo"], trace["p2_elo"])
for p in (fixture.p1_win_probability, leaky_elo_probability(fixture, label=1)):
    print(round(p, 4), round(-log(p), 4))
```

Predict which calculation will score better. The only extra piece of information in the second forecast is its own winner.

```bash
.venv/bin/python "$work/leak.py"
.venv/bin/python -m acebet.dataprep.historical_replay "$work/matches.parquet"
```

The comparison scores the 1,487 eligible matches before September 1.
The as-of replay gives log loss about **0.6741**. The deliberately leaky
calculation first applies each match's own winner to its ratings and
then scores the forecast. Its log loss falls to **0.5893**. It gets a
better score because the update moves probability toward the known winner.
The repair predicts from the cutoff state and applies a result only when
it becomes eligible for later forecasts. The archive records settled fixtures. Its result delay and the assumption that surface was known at the cutoff allow this retrospective comparison. Verifying a live forecast would require observed publication times and schedule coverage.

To test the correction yourself, open `test_cutoff_ignores_late_corrections_future_matches_and_current_winner` in `tests/test_historical_pipeline.py`. Move the correction from 30 minutes after noon to one minute before noon. Predict the rating that will break the original assertion, rerun the cutoff test, then restore the timestamp. You now have a time-ordered input for [training and validation](./train-under-deployment-conditions.md).
