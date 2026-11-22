# Course acceptance checks

**Orientation (pending human walkthrough).** Give a new reader the course homepage and ten minutes. Ask who the forecast is for, which information can enter it at noon, what gets trained, how batch and HTTP replies can be compared, and which evidence would allow a release. Record the date, revision, elapsed time, answers and any point where the reader needed help.

**Blind diagnosis (pending human walkthrough).** In a disposable copy, find `results_as_of` in
`src/acebet/dataprep/historical_replay.py` and replace the check
`result.available_at <= cutoff` with `result.event_date <= cutoff.date()`.
Build a new candidate with `just forecast-candidate` and find fixture `425753`
in `bundle/final/forecasts.csv`. Its Elo gap changes from about `-127.35` to
`-123.80` while it still has a plausible probability. Give the learner the
modified copy, that forecast, and the learner brief. Withhold the original
results, the change, and its location. A diagnosis should show why a result
from an earlier event can arrive after the cutoff and why its label must be
excluded. Use the cutoff regression in `tests/test_historical_pipeline.py::test_cutoff_ignores_late_corrections_future_matches_and_current_winner` to verify the repair.

**Record the result.** Capture the date, checkout revision, fixture ID,
the learner's affected claim, the failing command and observation, the repair
diff, and the passing regression output. Ask the learner which evidence led
to the cause. Record that independently of CI. Status is pending until a
human learner completes the walkthrough.
