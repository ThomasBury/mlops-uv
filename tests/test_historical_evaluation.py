from dataclasses import replace
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

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
    assert selection["best_params"] == {"model__C": 0.3}
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

    def unexpected_fit(*args, **kwargs):
        pytest.fail("Final evaluation tried to fit a pipeline.")

    monkeypatch.setattr(Pipeline, "fit", unexpected_fit)
    altered = [*history[:-1], replace(history[-1], label=1 - history[-1].label)]
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
    elo = report["log_loss_delta_candidate_minus_baseline"]["elo"]
    assert elo["interval_95"][0] < 0 < elo["interval_95"][1]
    assert elo["weeks"] == 9
    assert (
        sum(bin_["matches"] for bin_ in report["calibration"]["logistic_tuned"]) == 369
    )
    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    assert (
        client.get_run(report["run_id"]).data.tags["selection_run_id"]
        == selection["run_id"]
    )
    assert {entry.path for entry in client.list_artifacts(selection["run_id"])} == {
        "membership.csv",
        "model.joblib",
        "search.csv",
        "selection.json",
    }
    assert {entry.path for entry in client.list_artifacts(report["run_id"])} == {
        "evaluation.json",
        "forecasts.csv",
    }
