# M4b. Track the study and recover a fitted candidate

Trial 11 in [M4a](m4a.md) tells you which `C` scored best. A colleague has the run ID and wants to reproduce F01. They need the fitted scaler, encoder and classifier, along with the record of which trial selected the setting.

## Track the trials and freeze the refit

Continue in the same Bash shell at the repository root, with `results`, `m4`, and the `python4` array from M4a.

If you are following the versioned checkpoints, run `git switch --detach course-v1-m4b` in your clean course checkout before training.

```bash
bundle="$m4/bundle"
"${python4[@]}" -m acebet.train.historical_evaluation train "$results" "$bundle"
```

The tracked command repeats the same seeded study from M4a. It opens a parent `selection` run and a child run inside each trial. The parent records the data and fold fingerprints, source hashes, dependency versions and search configuration. Each child records its trial number, `C`, the three fold losses, mean loss and run status. An exception inside a trial marks the child and parent failed and stops the build. [Tracking implementation](../src/acebet/train/historical_evaluation.py) · [MLflow Tracking API](https://mlflow.org/docs/latest/ml/tracking/tracking-api/)

After the study, `train_candidate` uses the winning `C` for a fresh fit on the 1,487 development matches with labels available by August 31 at 23:59:59 UTC.

```python
decision, trials = _optimize(frame, folds, experiment_id=run.info.experiment_id)
metrics["logistic_tuned"] = decision["best_mean_log_loss"]
model = logistic_pipeline(decision["best_params"]["model__C"]).fit(
    frame[FEATURES], frame["label"]
)
```

MLflow logs that pipeline as a scikit-learn model with a `models:/…` URI and a tag pointing to the winning child. The run also keeps the serving bundle. It contains `model.joblib`, `selection.json`, `membership.csv`, and `search.csv`. The model URI loads the fitted pipeline in MLflow. The serving bundle carries the exact Joblib bytes identified by the forecast response and the frozen record used in evaluation.

In another terminal, set `m4` to the absolute scratch path, then open the local experiment. Stop the server when done.

```bash
uv run --locked --python 3.12 --group mlops mlflow ui \
    --backend-store-uri "sqlite:///$m4/mlflow.db" --no-serve-artifacts \
    --host 127.0.0.1 --port 5000
```

At `http://127.0.0.1:5000`, open `acebet-historical-forecast` and expand the `selection` run. Compare the winners and losers using `mean_log_loss` and the three fold metrics. Trial 5's `C` was about `0.00139`. Use the child scores to identify where it lost the most to trial 11. Find the logged model's winning-run tag and use the run ID to open that child's metrics.

## Recover into an empty directory

The parent run ID is in `selection.json`. Keep the tracking database and its artifact directory where they were created. Load only artifacts you trust, since Joblib can execute code during deserialization. The script below recovers the run's files into a new bundle and downloads the logged MLflow model into a separate scratch directory. It blocks `Pipeline.fit` while loading and replaying F01.

```bash
recovered="$m4/recovered"
"${python4[@]}" - "$m4" "$bundle/selection.json" "$recovered" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys
from unittest.mock import patch

import joblib
import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

scratch, original_record, recovered = map(Path, sys.argv[1:])
study = json.loads((scratch / "study.json").read_text())
run_id = json.loads(original_record.read_text())["run_id"]
tracking_uri = f"sqlite:///{scratch.resolve() / 'mlflow.db'}"
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri=tracking_uri)

parent = client.get_run(run_id)
assert parent.info.status == "FINISHED"
recovered.mkdir()
for name in ("selection.json", "model.joblib", "membership.csv", "search.csv"):
    client.download_artifacts(run_id, name, str(recovered))

selection = json.loads((recovered / "selection.json").read_text())
assert selection["run_id"] == run_id
for name in ("best_trial_number", "best_params", "fold_membership_sha256", "config"):
    assert selection[name] == study[name], name
for name, expected in selection["artifacts_sha256"].items():
    assert sha256((recovered / name).read_bytes()).hexdigest() == expected, name
assert parent.data.params["model_sha256"] == selection["artifacts_sha256"]["model.joblib"]

children = client.search_runs(
    [parent.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
)
assert len(children) == study["config"]["search_trials"]
winner = client.get_run(selection["best_trial_run_id"])
assert winner.info.run_id in {child.info.run_id for child in children}
assert winner.info.status == "FINISHED"
assert int(winner.data.params["trial_number"]) == study["best_trial_number"]
np.testing.assert_allclose(
    winner.data.metrics["mean_log_loss"], study["best_mean_log_loss"], atol=1e-8, rtol=0,
)
tracked_trials = pd.read_csv(recovered / "search.csv")
pd.testing.assert_frame_equal(tracked_trials, pd.DataFrame(study["trials"]), atol=1e-8, rtol=0)

fixture = selection["fixture"]
gap = fixture["p1_elo"] - fixture["p2_elo"]
inputs = pd.DataFrame({"elo_gap": [gap, gap], "surface": [fixture["surface"], "Unseen"]})
expected = [fixture["logistic_p1_win_probability"], fixture["unseen_surface_p1_win_probability"]]
destination = scratch / "mlflow-recovery"
destination.mkdir()
with patch.object(Pipeline, "fit", side_effect=AssertionError("Recovery attempted to fit")):
    logged = mlflow.sklearn.load_model(selection["logged_model_uri"], dst_path=str(destination))
    serving = joblib.load(recovered / "model.joblib")
    for model in (logged, serving):
        np.testing.assert_allclose(model.predict_proba(inputs)[:, 1], expected, atol=1e-8, rtol=0)
    encoder = serving.named_steps["features"].named_transformers_["surface"]
    print(encoder.categories_[0], encoder.transform(inputs.iloc[[1]][["surface"]]).toarray())
print("Recovered", run_id, "from winning child", winner.info.run_id, "with F01", expected[0])
PY

bundle="$recovered"
```

F01 on Clay gives about `0.475945`. The surface encoder maps the unseen surface to three zeros, so the other features and intercept still produce a probability of about `0.480347`. That output exercises the pipeline directly. The forecast service checks the requested surface against the fixture.

Keep the canonical Parquet, the parent and child run IDs, and the recovered bundle. [M4c](m4c.md) opens the held-out period and scores those recovered model bytes.
