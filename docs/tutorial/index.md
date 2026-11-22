# Lessons { #tutorial }

Start with basic Python and an understanding of fitting a classifier. You can do the first lesson with a pencil. From lesson 2 onward, keep the same checkout and scratch directory. Use Python 3.12, Git, uv and a Unix-style shell. You will need curl for HTTP, Docker Engine with Buildx for the runtime exercise (or native Podman on Linux), and just for the final automation.

Follow Alex and Robin in **F01**, a small synthetic fixture whose inputs you can check by hand. The 2021 archive supplies the training population and a second, real replay. Each lesson hands something to the next.

## Read in order { #read-in-order }

| Lesson | What you start with and what you make | How to check it |
| --- | --- | --- |
| [1. Define a pre-match probability](./forecast-contract.md) | A reader names two players before play. Decide which outcome the number describes and when you can score it. | Classify five fixtures and expose the winner-first shortcut. |
| [2. Put the forecast in a Python package](./reproduce-the-fixture.md) | Turn the hand calculation into a command another Python user can import and run. | Install outside the source tree and reverse 0.70 to 0.30. |
| [3. Rebuild the match dataset](./rebuild-match-history.md) | Turn a pinned results CSV into identified, validated Parquet rows. | Recover 1,856 matches twice and challenge a duplicate ID. |
| [4. Build features from what was known](./build-point-in-time-features.md) | Reconstruct the ratings available when the reader asked. | Compute F01 by hand, admit its winner early, then try a late correction. |
| [5. Train under deployment conditions](./train-under-deployment-conditions.md) | Choose a validation population and fit preprocessing on each earlier window. | Inspect dates and compare the scaler means 2.694 and 4.152. |
| [6. Freeze a candidate and judge the evidence](./evaluate-a-forecast.md) | Use the development folds to select one setting and freeze its fitted model. | Recompute paired loss for 369 fixtures and interpret nine weeks of evidence. |
| [7. Serve the frozen forecast](./serve-the-frozen-forecast.md) | Give offline and HTTP callers the same checked model and history. | Compare two replies and diagnose a live process that cannot forecast. |
| [8. Reproduce the service on another machine](./reproduce-the-forecast-runtime.md) | Put the fitted pipeline and its input history inside a runtime image. | Build from your own bundle and replay both fixtures without host mounts. |
| [9. Investigate a forecast after serving](./diagnose-and-roll-back-a-forecast.md) | Trace one request, then join recorded forecasts to results as they arrive. | Locate a delay and advance the clock for fixture 424816. |
| [10. Decide what a scheduled run may change](./automate-and-audit-a-forecast.md) | Apply the release rule, restore a known pair of assets and automate the checks. | Explain a rejected promotion and independently diagnose an unfamiliar forecast. |

Use the [historical command and artifact reference](../reference/historical-forecast.md) to look up a schema, inspect a failed run or repeat the full numerical comparison. The [packaged quickstart](../getting-started.md) reproduces a separate 2018 prediction.
