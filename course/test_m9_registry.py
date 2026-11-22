"""Verify that registry decisions and loaded forecast assets agree when activated."""

from contextlib import contextmanager
from math import isclose
from pathlib import Path

from fastapi.testclient import TestClient
import mlflow
import pytest

from acebet.app.historical import create_app
from acebet.dataprep.historical_replay import prior_day_cutoff
from acebet.dataprep.historical_results import build_results, read_results
from acebet.historical_forecast import ForecastRequest, load_forecaster
from acebet.model_artifacts import read_json, sha256_file
from acebet.train.historical_evaluation import (
    REPLAY_ATOL,
    evaluate_candidate,
    train_candidate,
)
from m9_bundle import open_bundle
from m9_registry import (
    ALIAS,
    PROD,
    promote,
    register,
    registry,
    review,
    select_champion,
    stage,
)
from m9_rehearsal import delayed_run, synthetic_scores

SNAPSHOT = Path(__file__).resolve().parents[1] / "data/raw/valuebetennis-2021-atp.csv"


@pytest.fixture
def tracked_runs(tmp_path):
    genuine = tmp_path / "archive"
    genuine.mkdir()
    history = genuine / "matches.parquet"
    build_results(SNAPSHOT, history)
    results = read_results(history)
    incumbent = tmp_path / "incumbent"
    previous_uri = mlflow.get_tracking_uri()
    try:
        train_candidate(results, genuine / "bundle")
        evaluate_candidate(results, genuine / "bundle")
        delayed_run(history, incumbent)
    finally:
        mlflow.set_tracking_uri(previous_uri)
    return genuine, incumbent


def test_registry_promotion_activation_and_rollback(
    tracked_runs, tmp_path, monkeypatch
):
    genuine, incumbent = tracked_runs
    client = registry(tmp_path / "registry.db")

    def champion():
        return client.get_model_version_by_alias(PROD, ALIAS).version

    @contextmanager
    def serving(release):
        with monkeypatch.context() as env:
            env.setenv("ACEBET_FORECAST_RESULTS", str(release / "matches.parquet"))
            env.setenv("ACEBET_FORECAST_BUNDLE", str(release))
            with TestClient(create_app()) as service:
                assert service.get("/readyz").status_code == 200
                yield service

    match = next(
        row
        for row in read_results(genuine / "matches.parquet")
        if row.match_id == "424816"
    )
    request = ForecastRequest(
        fixture_id=match.match_id,
        p1_id=match.p1_id,
        p2_id=match.p2_id,
        surface=match.surface,
        cutoff=prior_day_cutoff(match.event_date),
    )

    def forecast(service, release):
        response = service.post("/forecast", json=request.model_dump(mode="json"))
        response.raise_for_status()
        reply = response.json()
        manifest = read_json(release / "release.json")
        assert reply["model_sha256"] == manifest["model_sha256"]
        assert reply["history_sha256"] == manifest["history_sha256"]
        offline = load_forecaster(release / "matches.parquet", release).predict(request)
        assert isclose(
            reply["p1_win_probability"],
            offline.p1_win_probability,
            abs_tol=REPLAY_ATOL,
            rel_tol=0,
        )
        return reply

    incumbent_dev = register(client, incumbent)["version"]
    incumbent_rehearsal = tmp_path / "incumbent-rehearsal"
    synthetic_scores(incumbent, incumbent_rehearsal)
    incumbent_review = review(client, incumbent_dev, incumbent_rehearsal)
    before = promote(client, incumbent_review, incumbent_rehearsal)["production"][
        "version"
    ]
    before_release = tmp_path / "before"
    stage(client, before, incumbent / "matches.parquet", before_release)

    with serving(before_release) as still_active:
        original = forecast(still_active, before_release)
        candidate_dev = register(client, genuine)["version"]
        rejection = review(client, candidate_dev, genuine)
        assert rejection["rehearsal"] is None and rejection["gate"]["accepted"] is False
        with pytest.raises(ValueError, match="retained the incumbent"):
            promote(client, rejection, genuine)
        assert champion() == before
        assert forecast(still_active, before_release) == original

        rehearsal = tmp_path / "candidate-rehearsal"
        synthetic_scores(genuine, rehearsal)
        approval = review(client, candidate_dev, rehearsal)
        assert approval["rehearsal"] and approval["gate"]["accepted"] is True

        newer_dev = register(client, incumbent)["version"]
        assert int(newer_dev) > int(candidate_dev)
        wrong_version = {
            **approval,
            "development": {**approval["development"], "version": newer_dev},
        }
        with pytest.raises(ValueError, match="does not match"):
            promote(client, wrong_version, rehearsal)
        with pytest.raises(ValueError, match="no longer agrees"):
            promote(client, {**approval, "evaluation_sha256": "0" * 64}, rehearsal)
        assert champion() == before

        approved = promote(client, approval, rehearsal)
        after = approved["production"]["version"]
        assert approved["source"] == f"models:/acebet-dev/{candidate_dev}"
        assert promote(client, approval, rehearsal) == approved
        assert champion() == after != before
        assert forecast(still_active, before_release) == original

        after_release = tmp_path / "after"
        with pytest.raises(ValueError, match="Canonical results differ"):
            stage(client, after, incumbent / "matches.parquet", after_release)
        assert not after_release.exists()
        manifest = stage(client, after, genuine / "matches.parquet", after_release)
        assert manifest["production"]["version"] == after
        assert manifest["model_sha256"] == sha256_file(genuine / "bundle/model.joblib")

    with serving(after_release) as activated:
        changed = forecast(activated, after_release)
        assert changed["model_sha256"] != original["model_sha256"]
        assert changed["history_sha256"] != original["history_sha256"]
        assert changed["feature_state_sha256"] != original["feature_state_sha256"]

    with serving(before_release) as restored:
        assert forecast(restored, before_release) == original
    select_champion(client, before)
    assert champion() == before


def test_packaged_registry_relocates_tracked_artifacts(tmp_path):
    packed = Path(__file__).with_name("bundles") / "m9a-prerequisites.tar.gz"
    installed = tmp_path / "new checkout + lab"
    open_bundle(packed, installed)
    client = registry(installed / "registry.db")
    incumbent = read_json(installed / "before/release.json")
    assert (
        client.get_model_version_by_alias(PROD, ALIAS).version
        == incumbent["production"]["version"]
    )
    assert (
        stage(
            client,
            incumbent["production"]["version"],
            installed / "incumbent/matches.parquet",
            installed / "restaged",
        )
        == incumbent
    )

    archive = installed / "archive"
    dev = register(client, archive)
    selection = read_json(archive / "bundle/selection.json")
    tracker = registry(archive / "mlflow.db")
    model_id = selection["logged_model_uri"].removeprefix("models:/")
    artifact_uri = tracker.get_logged_model(model_id).artifact_location
    assert artifact_uri.startswith(archive.as_uri() + "/")
    decision = review(client, dev["version"], archive)
    assert decision["rehearsal"] is None
    assert decision["gate"]["accepted"] is False
    with pytest.raises(FileExistsError):
        open_bundle(packed, installed)
