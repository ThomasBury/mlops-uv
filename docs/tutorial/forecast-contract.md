# 1. Define a pre-match probability

A tennis reader comparing two named players before a match wants the
probability that the first player they name wins. If Alex Silva has a 70%
chance, the reader should still expect Alex to lose some matches with forecasts
like this one. A betting decision would also require prices, costs, and a rule
for deciding when to act.

The five fixtures below are synthetic. In the later 2021 replay, the archive contains settled results. A result’s release time has to be assumed.

## Define the row and the result

- **Fixture.** One row in an evaluation run represents one scheduled match at
  one recorded cutoff. Use a stable fixture ID, the two distinct stable
  player IDs, and the cutoff to identify the example. Fix one cutoff per
  fixture before evaluating so repeated predictions for a match count once.
- **Scope.** At the cutoff, forecast men's singles main-draw matches at ATP
  Tour events that are still scheduled and have two known players. The
  probability is conditional on the match being played to completion. A
  walkover means no play begins. A
  retirement means play begins and a player stops before the match is
  completed. Exclude both, along with cancellations, from result scoring.
  Keep their status so the scoring denominator can be checked.
- **Player order.** Put the smaller stable player ID first in the evaluation
  row, comparing IDs as strings. Keep each player's inputs attached to their
  ID when ordering the pair. The label is `1` if that first player wins a
  completed match and `0` if the other wins. For the reader, express the
  probability in the requested order. If the reader names the second player
  first, return the complement, `1 - p`.
- **Result availability.** A completed match can be scored once its official
  winner is available to the evaluator. Until the result or a terminal
  exclusion is available, the forecast is pending and has no usable label.
  A published walkover or retirement closes the fixture without a label under
  this contract. A pending or excluded fixture never receives a `0` label.

## Decide what the forecast may know

Choose and record a UTC cutoff before play, without using the outcome to pick
it. A live forecast must also be issued before play. An input is eligible if
its specific value, including any revision, was available by the cutoff.
Permitted inputs include the announced surface, as-of player rankings, and
ratings derived from earlier results already published at the cutoff. The
current match's result, score, in-play events, and later corrections are
outside that boundary. If a record has only a date, use completed earlier
periods with evidence or a documented assumption about when the source made
the records available. A match date alone does not establish availability.

## Try the contract on five fixtures

Alex was ranked 30 at noon. At 12:30 the feed corrected his rank to 28.
Which value belongs in the noon forecast? Write it down before opening the answer key.

Every fixture below was listed before its cutoff as an in-scope match on
clay between Alex Silva (`P01`) and Robin Novak (`P02`). For `F01`, read this
timeline. All times are UTC on 2030-06-10 unless stated otherwise.

<figure class="course-figure">
<ol class="course-flow">
<li><strong>June 9, 18:00</strong><br>Clay announced. Alex ranked 30.</li>
<li><strong>11:00</strong><br>Robin’s earlier result arrives.</li>
<li><strong>12:00 cutoff</strong><br>Record the forecast for Alex.</li>
<li><strong>12:30</strong><br>Alex’s rank is corrected to 28.</li>
<li><strong>13:00–14:50</strong><br>Play begins and ends.</li>
<li><strong>15:10</strong><br>The official winner becomes observable.</li>
</ol>
<figcaption>F01 on June 10, 2030 (UTC). Read up to noon to make the forecast. Read up to the reporting time to decide whether you can score it.</figcaption>
</figure>

The other forecasts use a 12:00 UTC cutoff on their respective match dates.
Read each case at 16:00 UTC on that date. Each status shown was published
after its cutoff if the fixture had reached a result or exclusion.

| Fixture | Date | Information at 16:00 UTC |
| --- | --- | --- |
| `F01` | June 10 | Completed. Alex won. |
| `F02` | June 12 | Completed. Robin won. |
| `F03` | June 14 | Alex advanced by walkover. |
| `F04` | June 16 | Robin advanced when Alex retired after play began. |
| `F05` | June 18 | Play began at 13:00. No result or terminal status is available yet. |

For each fixture, record whether it qualified for forecasting at noon and
whether you can score it at 16:00. For `F01`, name which timeline inputs you
can use, and decide whether you can score it at 15:00. Then write the labels
for `F01` and `F02` twice. First copy a results feed that puts the winner in
the first player slot. Then use stable player order. What could a model
learn from the first version without learning anything about tennis?

??? success "Answer key"

    All five fixtures qualified for a forecast at their cutoffs. At 16:00,
    `F01` and `F02` are scorable, with labels `1` and `0`. `F03` and `F04`
    are excluded. `F05` remains pending. At 15:00, `F01` is still pending
    because the official result has yet to be published.

    At the `F01` cutoff, use the announced clay surface, Alex's ranking of
    30, and Robin's published earlier result or a feature derived from it.
    The ranking of 28, events during `F01`, and its result all became
    available after noon. Editing the published winner later cannot change
    those eligible inputs or put Robin first in the evaluation row.

    Winner-first order gives Alex then Robin for `F01`, Robin then Alex for
    `F02`, and labels `1, 1`. Always predicting that the first player wins
    looks perfect on those rows. The player's slot already encodes the
    result. Stable order gives Alex then Robin for both rows and recovers
    labels `1, 0`.

## Decide what counts as success

Start with a **baseline**, a simple forecast to compare against. Use `0.50`
for either player in this exercise. Later, estimate the constant baseline
from the fraction of first-player wins in the fitting period, then score it
on the same later fixtures as the model.

**Probability quality.** For each scorable match, log loss is the negative
natural logarithm of the probability assigned to the outcome that happened.
Average it across the scorable matches. Lower is better. For `F01`, the
baseline assigns `0.50` to Alex's win. For `F02`, it assigns `1 - 0.50` to
Robin's win.

With a calculator, take the negative natural logarithm of 0.5. Both matches contribute that same loss.

Expect `0.6931`. Use **calibration** as a diagnostic. Among many forecasts
near 70%, check whether roughly 70% of the corresponding first players won.
Report how many scored matches contributed to each group. Two examples can
verify the calculation and ordering. Estimating calibration requires more data.

**Software success.** Replaying `F01` from the same cutoff and inputs yields
the same ordered IDs and probability. Reversing the reader's order returns
the complement. Check this with an illustrative `0.70` for Alex, which
becomes `0.30` when the reader names Robin first. Changing any post-cutoff
input or result leaves the forecast unchanged. Only completed matches with
published winners enter the score.

**Operational success.** Track which scheduled, in-scope fixtures received
a forecast by their planned cutoff. For matches that begin, verify that the
recorded issue time precedes the first point. Report unknown start times
separately. Retain the fixture ID and cutoff so each forecast can be matched
to its result. Report counts of scored, pending, and excluded forecasts
alongside average loss so the reader can see how much of the forecast
period the score covers.

**Finish when** a second reader can reproduce your eligibility, input, and
scoring decisions before opening the answer key. Carry `F01`, its noon UTC
cutoff, and its `0.50` baseline into the [reproducible Python exercise](./reproduce-the-fixture.md).
