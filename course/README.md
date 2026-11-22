# AceBet MLOps course

Build a tennis forecast you can reproduce from its match history, then follow the same model into a service and through a release. Start at M0 for the full course. You can also start at M5 with a frozen forecast or at M9a with a seeded registry.

## Choose a checkpoint

The maintainers publish the `course-v1-*` tags after the course lands on the default branch. The final checkpoint, `course-v1-m10`, includes this index and all the lesson materials. Until the tags are published, you can start [M5](m5.md) or [M9a](m9a.md) directly from a checkout that contains this course and its archives.

Once the tags are available, open the checkpoint for your lesson in a separate clone. Run these commands from a repository that has the tags to open M5.

```bash
practice="$(mktemp -d)"
git clone . "$practice/acebet"
cd "$practice/acebet"
git switch --detach course-v1-m5
```

Keep the lesson's scratch files outside that clone. If a later lesson uses variables from an earlier one, keep the same Bash session or set those paths again. The tag names use lowercase letters for the split milestones.

| Milestone | Lesson | Start with |
| --- | --- | --- |
| M0 | [Contract and installable prototype](m0.md) | Bash, uv and Python 3.12 |
| M1 | [Rebuild the match dataset](m1.md) | The pinned CSV shipped in the checkout |
| M2 | [Build features as of the cutoff](m2.md) | M1's Parquet or the pinned CSV |
| M3 | [Design a credible comparison](m3.md) | M1 or M2's Parquet, or the pinned CSV |
| M4a | [Tune regularization](m4a.md) | M3's comparison or the pinned CSV |
| M4b | [Track and recover a fitted candidate](m4b.md) | M4a's study and history |
| M4c | [Judge the frozen candidate](m4c.md) | M4b's recovered model and history |
| M5 | [Serve and package the frozen forecast](m5.md) | M4c's evaluated bundle or the [serving prerequisite](bundles/m5-prerequisites.tar.gz). HTTP, concurrency and containers are in this lab. |
| M6 | [Diagnose a forecast and rehearse rollback](m6.md) | M5's requests, model, history and image |
| M7 | [Automate and audit the forecast](m7.md) | The pinned CSV, `just` and a facilitator for the unfamiliar case |
| M8 | [Investigate forecasts on a replay clock](m8.md) | The supplied [incident capture](fixtures/m8-incident.jsonl) |
| M9a | [Promote and roll back with MLflow](m9a.md) | The [registry prerequisite](bundles/m9a-prerequisites.tar.gz), with the incumbent already staged |
| M9b | [Automate candidate review](m9b.md) | M9a's release or the standalone release test |
| M10 | [Predict delivery delays at booking](m10.md) | The supplied starter and two synthetic CSVs |

To open any row, prefix its lowercase milestone with `course-v1-`. Use [M7 facilitator notes](m7-facilitator.md) to prepare the audit case. Open the [M10 worked solution](m10-solution.md) after you have tried the exercise.

The archives contain fitted teaching models. Load them from a trusted course checkout and use the locked dependencies in the lesson. [Archive checksums and rebuild instructions](bundles/README.md)
