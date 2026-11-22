from dataclasses import replace
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import fmean

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from acebet.dataprep.historical_replay import DEVELOPMENT_END, prior_day_cutoff
from acebet.dataprep.historical_results import (
    build_results,
    content_sha256,
    ingest_snapshot,
    normalize_results,
)
from acebet.train.historical_evaluation import (
    FEATURES,
    VALIDATION_STARTS,
    development_folds,
    development_results,
    evaluate_candidate,
    logistic_pipeline,
    main,
    paired_week_interval,
    replay_frame,
    train_candidate,
    tune_development,
)

SNAPSHOT = Path(__file__).resolve().parents[1] / "data/raw/valuebetennis-2021-atp.csv"


@pytest.fixture(scope="module")
def history():
    return normalize_results(ingest_snapshot(SNAPSHOT))


def test_selection_excludes_late_labels_and_fits_preprocessing_on_each_training_window(
    history,
):
    early = history[0]
    late = replace(
        early,
        match_id="late-label",
        event_date=date(2021, 8, 31),
        available_at=datetime(2021, 9, 1, tzinfo=timezone.utc),
    )
    development = development_results([*history, late])
    assert len(development) == 1487
    assert late not in development
    assert all(
        row.available_at <= prior_day_cutoff(DEVELOPMENT_END) for row in development
    )
    frame = replay_frame(development)
    previous_train = set()
    all_validation = set()
    for start, (train_idx, valid_idx) in zip(
        VALIDATION_STARTS, development_folds(frame)
    ):
        train, valid = frame.iloc[train_idx], frame.iloc[valid_idx]
        assert (
            train["available_at"].max()
            <= prior_day_cutoff(start)
            <= valid["cutoff"].min()
        )
        assert previous_train < set(train["match_id"])
        assert set(train["match_id"]).isdisjoint(valid["match_id"])
        assert all_validation.isdisjoint(valid["match_id"])
        previous_train = set(train["match_id"])
        all_validation.update(valid["match_id"])

        model = logistic_pipeline().fit(train[FEATURES], train["label"])
        transformers = model.named_steps["features"].named_transformers_
        assert transformers["elo"].mean_[0] == pytest.approx(train["elo_gap"].mean())
        assert set(transformers["surface"].categories_[0]) == set(train["surface"])
        unseen = valid.iloc[[0]][FEATURES].assign(surface="Unseen")
        assert not transformers["surface"].transform(unseen[["surface"]]).nnz
        assert 0 < model.predict_proba(unseen)[0, 1] < 1


def test_compare_exports_temporal_membership_and_shuffled_counterexample(
    history, tmp_path, capsys
):
    results = tmp_path / "matches.parquet"
    build_results(SNAPSHOT, results)
    output = tmp_path / "comparison"
    args = ["compare", str(results), "--output", str(output)]
    main(args)
    report = json.loads(capsys.readouterr().out)
    assert report == json.loads((output / "comparison.json").read_text())
    assert report["data_sha256"] == content_sha256(history)
    assert report["development_sha256"] == content_sha256(development_results(history))
    assert report["final_start"] == "2021-09-01"

    membership = output / "membership.csv"
    assert (
        report["membership_sha256"]
        == hashlib.sha256(membership.read_bytes()).hexdigest()
    )
    rows = pd.read_csv(membership, dtype={"match_id": str})
    assert set(rows["split"]) == {"fold_1", "fold_2", "fold_3"}
    assert not rows.duplicated(["split", "match_id"]).any()
    assert set(rows["match_id"]) <= {
        row.match_id for row in development_results(history)
    }
    counts = rows.groupby(["split", "partition"]).size().unstack()
    assert counts["train"].is_monotonic_increasing
    assert len(report["fold_log_loss"]) == 3

    diagnostic = report["random_split_diagnostic"]
    by_id = {row.match_id: row for row in history}
    train = by_id[diagnostic["latest_training_match_id"]]
    validation = by_id[diagnostic["earliest_validation_match_id"]]
    cutoff = prior_day_cutoff(validation.event_date)
    assert train.available_at > cutoff
    assert (
        diagnostic["latest_training_label_available_at"]
        == train.available_at.isoformat()
    )
    assert diagnostic["earliest_validation_cutoff"] == cutoff.isoformat()

    with pytest.raises(FileExistsError):
        main(args)
    assert json.loads((output / "comparison.json").read_text()) == report


def test_paired_interval_counts_calendar_weeks_and_weights_matches():
    days = pd.Series([date(2021, 9, 5), date(2021, 9, 6), date(2021, 9, 12)])
    interval = paired_week_interval(np.array([1.0, -1.0, -1.0]), days)
    assert interval["mean_delta"] == pytest.approx(-1 / 3)
    assert interval["interval_95"] == [-1.0, 1.0]
    assert interval["weeks"] == 2


def test_frozen_evaluation_scores_only_final_fixtures_and_logs_both_runs(
    history, tmp_path, monkeypatch, request
):
    mlflow = pytest.importorskip("mlflow")
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", previous_uri)
    request.addfinalizer(lambda: mlflow.set_tracking_uri(previous_uri))
    bundle = tmp_path / "model"
    selection = train_candidate(history, bundle)
    assert selection["fixture"]["unseen_surface_encoding"] == [0, 0, 0]
    assert not (bundle / "final").exists()
    membership = pd.read_csv(bundle / "membership.csv", dtype={"match_id": str})
    random = membership.loc[membership["split"] == "random_diagnostic"]
    random_train = random.loc[random["partition"] == "train"].set_index("match_id")
    random_valid = random.loc[random["partition"] == "validation"].set_index("match_id")
    latest = random_train["available_at"].max()
    cutoffs = random_valid["cutoff"]
    assert (cutoffs < latest).sum() == 444
    diagnostic = selection["random_split_diagnostic"]
    assert (
        random_train.loc[diagnostic["latest_training_match_id"], "available_at"]
        == latest
    )
    assert cutoffs.loc[diagnostic["earliest_validation_match_id"]] == cutoffs.min()

    # Changing a held-out outcome can change the archive identity, but never a trial.
    altered = [*history[:-1], replace(history[-1], label=1 - history[-1].label)]
    study = tune_development(altered)
    assert study["data_sha256"] != selection["data_sha256"]
    assert study["development_sha256"] == selection["development_sha256"]
    assert study["fold_membership_sha256"] == selection["fold_membership_sha256"]
    pd.testing.assert_frame_equal(
        pd.read_csv(bundle / "search.csv"),
        pd.DataFrame(study["trials"]),
        rtol=0,
        atol=1e-8,
    )
    best = min(
        study["trials"], key=lambda row: (row["mean_log_loss"], row["trial_number"])
    )
    assert (
        selection["best_trial_number"]
        == study["best_trial_number"]
        == best["trial_number"]
    )
    assert selection["best_params"] == {"model__C": best["C"]}

    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    parent = client.get_run(selection["run_id"])
    children = client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
    )
    by_trial = {int(child.data.params["trial_number"]): child for child in children}
    assert len(children) == len(by_trial) == study["config"]["search_trials"]
    for trial in study["trials"]:
        child = by_trial[trial["trial_number"]]
        assert child.info.status == "FINISHED"
        assert float(child.data.params["C"]) == trial["C"]
        losses = [child.data.metrics[f"fold_{n}_log_loss"] for n in range(1, 4)]
        assert losses == [trial[f"fold_{n}_log_loss"] for n in range(1, 4)]
        assert (
            child.data.metrics["mean_log_loss"]
            == fmean(losses)
            == trial["mean_log_loss"]
        )
    winning_id = by_trial[best["trial_number"]].info.run_id
    assert (
        selection["best_trial_run_id"]
        == parent.data.tags["best_trial_run_id"]
        == winning_id
    )
    logged_model = client.get_logged_model(
        selection["logged_model_uri"].removeprefix("models:/")
    )
    assert logged_model.source_run_id == parent.info.run_id
    assert logged_model.tags["best_trial_run_id"] == winning_id

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Recovery or final evaluation tried to fit a pipeline.")

    monkeypatch.setattr(Pipeline, "fit", unexpected_fit)
    recovered = tmp_path / "recovered"
    recovered.mkdir()
    for name in ("selection.json", "model.joblib", "membership.csv", "search.csv"):
        client.download_artifacts(selection["run_id"], name, str(recovered))
        assert (recovered / name).read_bytes() == (bundle / name).read_bytes()
    bundle = recovered

    destination = tmp_path / "mlflow-recovery"
    destination.mkdir()
    model = mlflow.sklearn.load_model(
        selection["logged_model_uri"], dst_path=str(destination)
    )
    fixture = selection["fixture"]
    gap = fixture["p1_elo"] - fixture["p2_elo"]
    inputs = pd.DataFrame(
        {"elo_gap": [gap, gap], "surface": [fixture["surface"], "Unseen"]}
    )
    np.testing.assert_allclose(
        model.predict_proba(inputs)[:, 1],
        [
            fixture["logistic_p1_win_probability"],
            fixture["unseen_surface_p1_win_probability"],
        ],
        atol=1e-8,
        rtol=0,
    )

    with pytest.raises(ValueError, match="results differ"):
        evaluate_candidate(altered, bundle)
    model_path = bundle / "model.joblib"
    original = model_path.read_bytes()
    model_path.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="model.joblib differs"):
        evaluate_candidate(history, bundle)
    model_path.write_bytes(original)

    report = evaluate_candidate(history, bundle)
    with pytest.raises(FileExistsError):
        evaluate_candidate(history, bundle)
    forecasts = pd.read_csv(bundle / "final/forecasts.csv", dtype={"match_id": str})
    assert forecasts["match_id"].tolist() == [
        row.match_id for row in history if row.event_date >= DEVELOPMENT_END
    ]
    assert report["matches"] == 369
    assert report["log_loss"]["elo"] == pytest.approx(0.626088125)
    assert report["log_loss"]["logistic_tuned"] == pytest.approx(0.629485311, abs=1e-8)
    elo = report["log_loss_delta_candidate_minus_baseline"]["elo"]
    assert 0 < elo["interval_95"][0] < elo["mean_delta"] < elo["interval_95"][1]
    assert elo["weeks"] == 9
    assert (
        sum(bin_["matches"] for bin_ in report["calibration"]["logistic_tuned"]) == 369
    )
    assert (
        client.get_run(report["run_id"]).data.tags["selection_run_id"]
        == selection["run_id"]
    )
    assert {entry.path for entry in client.list_artifacts(selection["run_id"])} == {
        "membership.csv",
        "model.joblib",
        "search.csv",
        "selection.json",
        "study.json",
    }
    assert {entry.path for entry in client.list_artifacts(report["run_id"])} == {
        "evaluation.json",
        "forecasts.csv",
    }


def test_failed_trial_keeps_tracking_evidence_without_freezing_bundle(
    history, tmp_path, monkeypatch, request
):
    mlflow = pytest.importorskip("mlflow")
    previous_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", previous_uri)
    request.addfinalizer(lambda: mlflow.set_tracking_uri(previous_uri))
    original_fit = Pipeline.fit

    def fail_trial(self, *args, **kwargs):
        if self.named_steps["model"].C != 1.0:
            raise RuntimeError("trial fit failed")
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", fail_trial)
    bundle = tmp_path / "bundle"
    with pytest.raises(RuntimeError, match="trial fit failed"):
        train_candidate(history, bundle)
    assert not bundle.exists()

    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = client.get_experiment_by_name("acebet-historical-forecast")
    runs = {
        run.info.run_name: run for run in client.search_runs([experiment.experiment_id])
    }
    assert runs.keys() == {"selection", "trial-00"}
    parent, child = runs["selection"], runs["trial-00"]
    assert child.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert parent.info.status == child.info.status == "FAILED"
    assert {entry.path for entry in client.list_artifacts(parent.info.run_id)} == {
        "study.json"
    }
