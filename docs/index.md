# What could we know before the match? { .brand-title }

At noon a reader asks for Alex Silva’s chance of beating Robin Novak on clay. Robin’s last result arrived at 11:00. A correction arrives at 12:30. Which version belongs in the forecast a reader saw at noon?

Work that example by hand, then build a service that can reproduce its answer. AceBet uses a small synthetic fixture, **F01**, to make timing mistakes visible. You will fit a model on a named extract of 2021 tennis results and test it on later matches. At the end you must decide whether the evidence warrants replacing Elo.

<figure class="course-figure">
<ol class="course-flow">
<li>Define the forecast</li>
<li>Acquire results</li>
<li>Validate matches</li>
<li>Build features</li>
<li>Train</li>
<li>Evaluate<small>Return to training if development evidence warrants a change.</small></li>
<li>Track experiments</li>
<li>Package</li>
<li>Serve<small>Batch file or HTTP response</small></li>
<li>Monitor<small>Send quality and service evidence to maintenance.</small></li>
<li>Maintain and automate</li>
</ol>
<figcaption>Follow a forecast from definition to maintenance. The final evaluation is reserved for a frozen candidate.</figcaption>
</figure>

## Follow the evidence

| Stage | Input | What can go wrong | What you carry forward |
| --- | --- | --- | --- |
| Define the forecast | Two player IDs and a cutoff | Winner-first ordering reveals the answer. | A conditional probability for the first named player. |
| Acquire results | A dated provider export | A refreshed download can change the past. | Attributed, byte-identified 2021 CSV extract. |
| Validate matches | Raw scores, winners and IDs | A repeated or incomplete match can count twice or receive a false label. | One Parquet row per eligible completed match. |
| Build features | Results available at the cutoff | Today's winner can leak into today's ratings. | An Elo gap in stable player order and a surface. |
| Train | Mature labels and pre-match inputs | Future results can slip into a fit meant for earlier forecasts. | Prevalence, Elo and a fitted logistic pipeline. |
| Evaluate | Later fixtures shared by all candidates | A small score difference can be sampling noise. | Paired losses, calibration counts and a release criterion. |
| Track experiments | Split membership, settings, code and predictions | A score alone cannot identify what produced it. | Linked selection and final MLflow runs. |
| Package | Frozen model, selection record and history | A different environment or history changes the reply. | A non-root image with identified assets. |
| Serve | A recorded fixture and cutoff | Different inference paths can silently disagree. | Batch predictions and HTTP replies with matching identities. |
| Monitor | Requests, traces and delayed winners | Fast replies can conceal deteriorating probabilities. | Service timings today and scored cohorts as labels mature. |
| Maintain and automate | Monitoring evidence and a candidate run | A schedule can repeatedly promote a weak model. | A review record, an explicit decision and a rehearsed rollback. |

Here, modern ML engineering means you can rebuild the named dataset, apply the same fitted transformations to the next match, and identify which model and feature state answered a request. You can test each claim with two players before you need any infrastructure.

[Start with the five fixtures](tutorial/forecast-contract.md#try-the-contract-on-five-fixtures) · [See all ten lessons and their checks](tutorial/index.md)

Looking for the working Fognini–Jarry example? The packaged predictor comes with its own prepared data and authenticated API. [Run the packaged quickstart](getting-started.md)
