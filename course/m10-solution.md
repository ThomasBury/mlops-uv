# M10 worked solution

Use this page after attempting [M10](m10.md). The [reference implementation](m10_solution.py) and independent checker run from the repository root.

```bash
report="$(mktemp)"
py=(python3.12)
# If you use uv to supply Python, use py=(uv run --no-project --python 3.12 python).
"${py[@]}" course/m10_solution.py course/fixtures/m10 2030-01-08 2030-01-16 > "$report"
"${py[@]}" course/m10_check.py course/m10_solution.py "$report"
```

## Two clocks determine what the forecast can use

One prediction estimates the probability that the parcel being booked will be delivered after its promised time. `P01` was delivered exactly at the promise and has label `0` from January 2 at 18:00 UTC. `P02` has label `1` from January 4 at 20:00 UTC. `P12` is unresolved. The archive records no outcome to score.

At noon on January 8, `P08` had the `P05` on-track scan as its newest eligible North depot event. The `P07` warning occurred the previous afternoon and was received at 14:00 UTC on January 8, two hours after the booking. Filtering on `event.occurred_at` let the starter use that warning. The repair filters on `event.available_at` and then selects the latest eligible scan by occurrence time. The checker changes the warning's status to expose the leak. It changes the available `P05` status to prove the feature still uses depot history.

```python
and event.available_at <= booking.booked_at
```

## Compare the two forecasts

`P01` through `P05` have training labels at midnight UTC on January 8. The outcomes for `P06` and `P07` arrive later. They miss the training deadline, and their bookings precede the evaluation period. At the January 16 scoring time, four of the six evaluation-period bookings have labels. `P12` remains unresolved and `P13`'s outcome arrives the next day. `P14` is booked after the scoring time.

Three of five training parcels are late, giving baseline probability `0.60`. One of three parcels with feature `0` is late, giving candidate probability `0.40` after adding counts. Both parcels with feature `1` are late, giving `0.75`.

| Evaluation parcel | Observed label | Baseline probability | Repaired candidate probability |
| --- | --- | --- | --- |
| P08 | 1 | 0.60 | 0.40 |
| P09 | 1 | 0.60 | 0.75 |
| P10 | 0 | 0.60 | 0.40 |
| P11 | 1 | 0.60 | 0.75 |

The leaky feature raises `P08`'s probability from `0.40` to `0.75`. Because that parcel really was late, the leaky candidate appears to score better. Candidate mean log loss rises from about `0.34347` to `0.50062` after the repair. Baseline mean log loss is `0.61219`. The repaired paired difference is `−0.11157`.

One defensible decision is to keep the baseline and test the repaired feature on a longer later period. Four scored synthetic parcels establish that the code can make and score the comparison. A stronger evaluation would use multiple periods with measured label coverage and depot workloads.

## Record an independent attempt

Give the learner the [lesson](m10.md), the starter, the checker, and the two CSVs. Keep this page and the reference implementation out of the materials until they submit their evidence. Record the elapsed time, any help needed, and which of the following checks passed.

- The written contract correctly classifies on-time, late, and unresolved parcels and states when a label can be used.
- The feature passes the availability probes and the learner explains the `P08` failure.
- The saved report passes the independent membership, probability, and loss checks.
- The written decision uses the repaired comparison and explains its limits.

If the learner reaches 60 minutes before finishing, record where they stopped and let them finish. Judge speed separately from whether they can apply the method.
