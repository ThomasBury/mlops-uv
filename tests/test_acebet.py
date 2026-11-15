import re
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from fastapi.testclient import TestClient

from acebet.app.main import app
from acebet.app.runtime import PredictionAssets
from acebet.app.dependencies.data_models import (
    PREDICTION_CONTEXT_ALLOWED_VALUES,
    ModelIdentity,
    PredictionRequest,
)
from acebet.app.dependencies.auth import (
    create_access_token,
    get_demo_users_db,
    get_password_hash,
    verify_password,
)
from acebet.model_artifacts import model_card_path, model_metadata_path
from acebet.settings import get_settings
from acebet.dataprep.dataprep import prepare_data
from acebet.app.dependencies.predict_winner import (
    make_online_prediction,
    resolve_player_snapshots,
)
from acebet.features import (
    MODEL_FEATURE_COLUMNS,
    align_features_to_model,
    build_latest_player_stats,
    build_match_feature_row,
    select_model_features,
)
from acebet.train.train import (
    build_model,
    export_model,
    get_lgb_params,
    prepare_data_for_training_clf,
    train_model,
)

pytestmark = pytest.mark.api


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    app.state.limiter.reset()
    yield
    app.state.limiter.reset()
    app.state.limiter.enabled = True


@pytest.fixture
def configured_auth_settings(monkeypatch):
    monkeypatch.setenv("ACEBET_SECRET_KEY", "test-signing-secret")
    monkeypatch.setenv("ACEBET_ENV", "test")
    monkeypatch.setenv("ACEBET_RATE_LIMITING_ENABLED", "false")
    monkeypatch.setenv("ACEBET_DEMO_USERNAME", "johndoe")
    monkeypatch.setenv("ACEBET_DEMO_PASSWORD", "secret")
    get_settings.cache_clear()
    get_demo_users_db.cache_clear()
    yield
    get_demo_users_db.cache_clear()
    get_settings.cache_clear()


@pytest.fixture
def client(configured_auth_settings):
    get_settings.cache_clear()
    get_demo_users_db.cache_clear()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def access_token(client: TestClient, configured_auth_settings) -> str:
    response = client.post(
        "/token",
        data={"username": "johndoe", "password": "secret"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture
def auth_headers(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


def get_prometheus_counter_value(
    metrics_text: str, metric_name: str, labels: str
) -> float:
    label_pattern = rf"\{{{labels}\}}" if labels else ""
    pattern = rf"^{re.escape(metric_name)}{label_pattern} ([0-9.eE+-]+)$"
    match = re.search(pattern, metrics_text, re.MULTILINE)
    if not match:
        return 0.0
    return float(match.group(1))


def test_login_for_access_token(client: TestClient, configured_auth_settings):
    response = client.post(
        "/token",
        data={"username": "johndoe", "password": "secret"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_token_request_logs_do_not_include_body_or_credentials(monkeypatch, tmp_path):
    log_path = tmp_path / "acebet.log"
    monkeypatch.setenv("ACEBET_SECRET_KEY", "test-signing-secret")
    monkeypatch.setenv("ACEBET_ENV", "test")
    monkeypatch.setenv("ACEBET_RATE_LIMITING_ENABLED", "false")
    monkeypatch.setenv("ACEBET_LOG_FORMAT", "json")
    monkeypatch.setenv("ACEBET_LOG_FILE", str(log_path))
    get_settings.cache_clear()
    get_demo_users_db.cache_clear()

    try:
        with TestClient(app) as log_client:
            response = log_client.post(
                "/token",
                data={"username": "johndoe", "password": "secret"},
            )
    finally:
        get_demo_users_db.cache_clear()
        get_settings.cache_clear()

    assert response.status_code == 200
    log_text = log_path.read_text(encoding="utf-8")
    assert "http_request_completed" in log_text
    assert "request_excerpt" not in log_text
    assert "username=johndoe" not in log_text
    assert "password=secret" not in log_text
    assert "johndoe" not in log_text
    assert "secret" not in log_text


def test_token_route_is_rate_limited(monkeypatch):
    monkeypatch.setenv("ACEBET_SECRET_KEY", "test-signing-secret")
    monkeypatch.setenv("ACEBET_ENV", "test")
    monkeypatch.setenv("ACEBET_RATE_LIMITING_ENABLED", "true")
    monkeypatch.setenv("ACEBET_TOKEN_RATE_LIMIT", "2/minute")
    get_settings.cache_clear()
    get_demo_users_db.cache_clear()

    try:
        with TestClient(app) as rate_limited_client:
            responses = [
                rate_limited_client.post(
                    "/token",
                    data={"username": "johndoe", "password": "secret"},
                )
                for _ in range(3)
            ]
    finally:
        get_demo_users_db.cache_clear()
        get_settings.cache_clear()

    assert [response.status_code for response in responses] == [200, 200, 429]


def test_operational_routes_are_not_rate_limited(monkeypatch):
    monkeypatch.setenv("ACEBET_SECRET_KEY", "test-signing-secret")
    monkeypatch.setenv("ACEBET_ENV", "test")
    monkeypatch.setenv("ACEBET_RATE_LIMITING_ENABLED", "true")
    monkeypatch.setenv("ACEBET_TOKEN_RATE_LIMIT", "1/minute")
    monkeypatch.setenv("ACEBET_PREDICTION_RATE_LIMIT", "1/minute")
    get_settings.cache_clear()
    get_demo_users_db.cache_clear()

    try:
        with TestClient(app) as rate_limited_client:
            responses = []
            for path in ["/healthz", "/readyz", "/metrics", "/docs", "/openapi.json"]:
                responses.extend(rate_limited_client.get(path) for _ in range(3))
    finally:
        get_demo_users_db.cache_clear()
        get_settings.cache_clear()

    assert all(response.status_code != 429 for response in responses)


def test_prediction_routes_share_user_rate_limit(monkeypatch):
    monkeypatch.setenv("ACEBET_SECRET_KEY", "test-signing-secret")
    monkeypatch.setenv("ACEBET_ENV", "test")
    monkeypatch.setenv("ACEBET_RATE_LIMITING_ENABLED", "true")
    monkeypatch.setenv("ACEBET_PREDICTION_RATE_LIMIT", "2/minute")
    get_settings.cache_clear()
    get_demo_users_db.cache_clear()

    payload = {
        "p1_name": "Fognini F.",
        "p2_name": "Jarry N.",
        "date": "2018-03-04",
        "series": "ATP250",
        "court": "Outdoor",
        "surface": "Hard",
        "round": "1st Round",
        "best_of": 3,
    }
    try:
        with TestClient(app) as rate_limited_client:
            token_response = rate_limited_client.post(
                "/token",
                data={"username": "johndoe", "password": "secret"},
            )
            assert token_response.status_code == 200
            headers = {
                "Authorization": f"Bearer {token_response.json()['access_token']}"
            }
            responses = [
                rate_limited_client.post(
                    "/predict/testing", headers=headers, json=payload
                ),
                rate_limited_client.post(
                    "/predict/production", headers=headers, json=payload
                ),
                rate_limited_client.post(
                    "/predict/testing", headers=headers, json=payload
                ),
            ]
    finally:
        get_demo_users_db.cache_clear()
        get_settings.cache_clear()

    assert [response.status_code for response in responses] == [200, 200, 429]


def test_healthz_returns_ok(client: TestClient, configured_auth_settings):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_returns_ready_when_assets_are_loaded(
    client: TestClient, configured_auth_settings
):
    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_readyz_returns_503_when_required_assets_are_missing(
    client: TestClient, configured_auth_settings
):
    client.app.state.runtime_assets.production = None

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_readyz_returns_503_when_testing_assets_are_missing_in_test_env(
    client: TestClient, configured_auth_settings
):
    client.app.state.runtime_assets.testing = None

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_metrics_endpoint_returns_prometheus_text(
    client: TestClient, configured_auth_settings
):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "acebet_http_requests_total" in body
    assert "acebet_prediction_requests_total" in body


def test_healthz_sets_request_id_response_header(
    client: TestClient, configured_auth_settings
):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]


def test_incoming_request_id_is_preserved(client: TestClient, configured_auth_settings):
    response = client.get("/healthz", headers={"X-Request-ID": "req-abc"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-abc"


def test_prediction_metrics_increment_for_testing_mode(
    client: TestClient,
    auth_headers: dict[str, str],
    configured_auth_settings,
):
    before = client.get("/metrics").text
    before_count = get_prometheus_counter_value(
        before,
        "acebet_prediction_requests_total",
        'mode="testing"',
    )

    response = client.post(
        "/predict/testing",
        headers=auth_headers,
        json={
            "p1_name": "Fognini F.",
            "p2_name": "Jarry N.",
            "date": "2018-03-04",
            "series": "ATP250",
            "court": "Outdoor",
            "surface": "Hard",
            "round": "1st Round",
            "best_of": 3,
        },
    )

    assert response.status_code == 200

    after = client.get("/metrics").text
    after_count = get_prometheus_counter_value(
        after,
        "acebet_prediction_requests_total",
        'mode="testing"',
    )

    assert after_count == before_count + 1.0


def test_sample_data_prediction_metric_increments_after_successful_testing_prediction(
    client: TestClient,
    auth_headers: dict[str, str],
    configured_auth_settings,
):
    before = client.get("/metrics").text
    before_count = get_prometheus_counter_value(
        before,
        "acebet_predicting_with_sample_data_total",
        "",
    )

    response = client.post(
        "/predict/testing",
        headers=auth_headers,
        json={
            "p1_name": "Fognini F.",
            "p2_name": "Jarry N.",
            "date": "2018-03-04",
            "series": "ATP250",
            "court": "Outdoor",
            "surface": "Hard",
            "round": "1st Round",
            "best_of": 3,
        },
    )

    assert response.status_code == 200

    after = client.get("/metrics").text
    after_count = get_prometheus_counter_value(
        after,
        "acebet_predicting_with_sample_data_total",
        "",
    )

    assert after_count == before_count + 1.0


def test_legacy_predict_route_no_longer_exists(
    client: TestClient,
    auth_headers: dict[str, str],
    configured_auth_settings,
):
    response = client.post(
        "/predict/",
        headers=auth_headers,
        json={
            "p1_name": "Fognini F.",
            "p2_name": "Jarry N.",
            "date": "2018-03-04",
            "series": "ATP250",
            "court": "Outdoor",
            "surface": "Hard",
            "round": "1st Round",
            "best_of": 3,
        },
    )

    assert response.status_code == 404


def test_rate_limit_demo_route_no_longer_exists(
    client: TestClient, configured_auth_settings
):
    response = client.get("/limit/?user_id=tester")

    assert response.status_code == 404


@pytest.mark.parametrize("path", ["/predict/production", "/predict/testing"])
def test_prediction_routes_reject_testing_field(
    path: str,
    client: TestClient,
    auth_headers: dict[str, str],
    configured_auth_settings,
):
    response = client.post(
        path,
        headers=auth_headers,
        json={
            "p1_name": "Fognini F.",
            "p2_name": "Jarry N.",
            "date": "2018-03-04",
            "series": "ATP250",
            "court": "Outdoor",
            "surface": "Hard",
            "round": "1st Round",
            "best_of": 3,
            "testing": False,
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize("path", ["/predict/production", "/predict/testing"])
def test_prediction_routes_reject_disabled_users(
    path: str,
    client: TestClient,
    configured_auth_settings,
):
    users_db = get_demo_users_db()
    users_db["johndoe"]["disabled"] = True
    token = create_access_token({"sub": "johndoe"})

    response = client.post(
        path,
        headers={"Authorization": f"Bearer {token}"},
        json={
            "p1_name": "Fognini F.",
            "p2_name": "Jarry N.",
            "date": "2018-03-04",
            "series": "ATP250",
            "court": "Outdoor",
            "surface": "Hard",
            "round": "1st Round",
            "best_of": 3,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Inactive user"


def test_testing_prediction_route_forbidden_in_production(monkeypatch):
    monkeypatch.setenv("ACEBET_SECRET_KEY", "test-signing-secret")
    monkeypatch.setenv("ACEBET_ENV", "production")
    monkeypatch.setenv("ACEBET_DEMO_PASSWORD", "production-password")
    get_settings.cache_clear()
    get_demo_users_db.cache_clear()

    try:
        with TestClient(app) as production_client:
            token_response = production_client.post(
                "/token",
                data={"username": "johndoe", "password": "production-password"},
            )
            assert token_response.status_code == 200
            access_token = token_response.json()["access_token"]
            assert production_client.app.state.runtime_assets.testing is None

            response = production_client.post(
                "/predict/testing",
                headers={"Authorization": f"Bearer {access_token}"},
                json={
                    "p1_name": "Fognini F.",
                    "p2_name": "Jarry N.",
                    "date": "2018-03-04",
                    "series": "ATP250",
                    "court": "Outdoor",
                    "surface": "Hard",
                    "round": "1st Round",
                    "best_of": 3,
                },
            )
    finally:
        get_demo_users_db.cache_clear()
        get_settings.cache_clear()

    assert response.status_code == 403


def test_readyz_in_production_does_not_require_testing_assets(monkeypatch):
    monkeypatch.setenv("ACEBET_SECRET_KEY", "test-signing-secret")
    monkeypatch.setenv("ACEBET_ENV", "production")
    monkeypatch.setenv("ACEBET_DEMO_PASSWORD", "production-password")
    get_settings.cache_clear()
    get_demo_users_db.cache_clear()

    try:
        with TestClient(app) as production_client:
            production_client.app.state.runtime_assets.testing = None
            response = production_client.get("/readyz")
    finally:
        get_demo_users_db.cache_clear()
        get_settings.cache_clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_bcrypt_helpers_support_seeded_and_new_hashes():
    seeded_hash = "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"

    assert verify_password("secret", seeded_hash)

    hashed_password = get_password_hash("another-secret")
    assert hashed_password.startswith("$2b$")
    assert verify_password("another-secret", hashed_password)
    assert not verify_password("wrong-password", hashed_password)


def test_read_users_me(
    client: TestClient, auth_headers: dict[str, str], configured_auth_settings
):
    response = client.get("/users/me", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["username"] == "johndoe"


def test_users_me_items_route_is_removed(
    client: TestClient, auth_headers: dict[str, str], configured_auth_settings
):
    response = client.get("/users/me/items/", headers=auth_headers)

    assert response.status_code == 404


def test_predict_match_outcome(
    client: TestClient, auth_headers: dict[str, str], configured_auth_settings
):
    response = client.post(
        "/predict/testing",
        headers=auth_headers,
        json={
            "p1_name": "Fognini F.",
            "p2_name": "Jarry N.",
            "date": "2018-03-04",
            "series": "ATP250",
            "court": "Outdoor",
            "surface": "Hard",
            "round": "1st Round",
            "best_of": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "predicted_winner",
        "predicted_class",
        "p1_win_probability",
        "prediction_mode",
        "model",
        "feature_freshness",
    }
    assert data["predicted_winner"] in {"Fognini F.", "Jarry N."}
    assert 0.0 <= data["p1_win_probability"] <= 1.0
    assert data["predicted_class"] in [0, 1]
    assert data["prediction_mode"] == "testing"
    assert data["feature_freshness"]["p1_age_days"] == 0
    assert data["feature_freshness"]["p2_age_days"] == 0


def test_predict_match_outcome_with_production_data(
    client: TestClient,
    auth_headers: dict[str, str],
    configured_auth_settings,
):
    client.app.state.runtime_assets.testing = None

    response = client.post(
        "/predict/production",
        headers=auth_headers,
        json={
            "p1_name": "Fognini F.",
            "p2_name": "Jarry N.",
            "date": "2018-03-04",
            "series": "ATP250",
            "court": "Outdoor",
            "surface": "Hard",
            "round": "1st Round",
            "best_of": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["prediction_mode"] == "production"
    assert 0.0 <= data["p1_win_probability"] <= 1.0
    assert data["predicted_class"] in [0, 1]
    assert set(data["model"]) == {"kind", "version", "sha256"}


def test_predict_match_outcome_for_future_match_uses_player_state(
    client: TestClient,
    auth_headers: dict[str, str],
    configured_auth_settings,
):
    response = client.post(
        "/predict/testing",
        headers=auth_headers,
        json={
            "p1_name": "Fognini F.",
            "p2_name": "Jarry N.",
            "date": "2030-01-01",
            "series": "ATP250",
            "court": "Outdoor",
            "surface": "Clay",
            "round": "The Final",
            "best_of": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_winner"] == (
        "Fognini F." if data["predicted_class"] == 1 else "Jarry N."
    )
    assert 0.0 <= data["p1_win_probability"] <= 1.0
    assert data["feature_freshness"]["p1_age_days"] > 0
    assert data["feature_freshness"]["p2_age_days"] > 0


def test_prepared_dataset_is_valid():
    df = prepare_data()

    assert not df.empty
    assert df["date"].is_monotonic_increasing
    assert "target" in df.columns
    assert "rank_diff" in df.columns
    assert "best_ranked" in df.columns


def test_latest_player_stats_uses_latest_player_appearance():
    df = prepare_data()
    stats = build_latest_player_stats(df)
    fognini_stats = stats.set_index("player").loc["Fognini F."]
    appearances = df[(df["p1"].eq("Fognini F.")) | (df["p2"].eq("Fognini F."))]

    assert "Fognini F." in set(stats["player"])
    assert fognini_stats["as_of_date"] == appearances["date"].max()


def test_build_match_feature_row_assembles_online_features():
    df = prepare_data()
    stats = build_latest_player_stats(df)
    p1_snapshot, p2_snapshot = resolve_player_snapshots(stats, "Fognini F.", "Jarry N.")
    features = build_match_feature_row(
        p1_name="Fognini F.",
        p2_name="Jarry N.",
        date="2030-01-01",
        p1_snapshot=p1_snapshot.feature_values(),
        p2_snapshot=p2_snapshot.feature_values(),
        series="ATP250",
        court="Outdoor",
        surface="Clay",
        round="The Final",
        best_of=3,
    )

    assert list(features.columns) == list(MODEL_FEATURE_COLUMNS)
    assert features.loc[0, "p1"] == "Fognini F."
    assert features.loc[0, "p2"] == "Jarry N."
    assert features.loc[0, "surface"] == "Clay"
    assert features.loc[0, "round"] == "The Final"
    assert features.loc[0, "year"] == 2030
    assert features.loc[0, "best_ranked"] in ["p1", "p2"]
    assert 0.0 <= features.loc[0, "proba_elo"] <= 1.0


def test_build_match_feature_row_ignores_player_stats_index_shape():
    df = prepare_data()
    stats = build_latest_player_stats(df)
    indexed_stats = stats.set_index("player", drop=False).sort_index()
    plain_snapshots = resolve_player_snapshots(stats, "Fognini F.", "Jarry N.")
    indexed_snapshots = resolve_player_snapshots(
        indexed_stats, "Fognini F.", "Jarry N."
    )

    from_plain_stats = build_match_feature_row(
        p1_name="Fognini F.",
        p2_name="Jarry N.",
        date="2030-01-01",
        p1_snapshot=plain_snapshots[0].feature_values(),
        p2_snapshot=plain_snapshots[1].feature_values(),
        series="ATP250",
        court="Outdoor",
        surface="Hard",
        round="1st Round",
        best_of=3,
    )
    from_indexed_stats = build_match_feature_row(
        p1_name="Fognini F.",
        p2_name="Jarry N.",
        date="2030-01-01",
        p1_snapshot=indexed_snapshots[0].feature_values(),
        p2_snapshot=indexed_snapshots[1].feature_values(),
        series="ATP250",
        court="Outdoor",
        surface="Hard",
        round="1st Round",
        best_of=3,
    )

    pd.testing.assert_frame_equal(from_plain_stats, from_indexed_stats)


def test_online_feature_contract_matches_training_feature_order():
    training_data = prepare_data_for_training_clf("2015-03-04", "2017-03-04")
    X, y = training_data.features, training_data.target
    model = build_model(get_lgb_params())
    model.fit(X, y)

    df = prepare_data()
    stats = build_latest_player_stats(df)
    p1_snapshot, p2_snapshot = resolve_player_snapshots(stats, "Fognini F.", "Jarry N.")
    online_features = build_match_feature_row(
        p1_name="Fognini F.",
        p2_name="Jarry N.",
        date="2030-01-01",
        p1_snapshot=p1_snapshot.feature_values(),
        p2_snapshot=p2_snapshot.feature_values(),
        series="ATP250",
        court="Outdoor",
        surface="Hard",
        round="1st Round",
        best_of=3,
    )

    aligned_training = align_features_to_model(model, X.head(1))
    aligned_online = align_features_to_model(model, online_features)
    probability, predicted_class = make_online_prediction(
        model=model,
        p1_name="Fognini F.",
        p2_name="Jarry N.",
        date="2030-01-01",
        p1_snapshot=p1_snapshot,
        p2_snapshot=p2_snapshot,
        series="ATP250",
        court="Outdoor",
        surface="Hard",
        round="1st Round",
        best_of=3,
    )

    assert list(aligned_training.columns) == list(X.columns)
    assert list(aligned_online.columns) == list(X.columns)
    assert 0.0 <= probability <= 1.0
    assert predicted_class in [0, 1]


def test_feature_contract_excludes_post_match_columns():
    training_data = prepare_data_for_training_clf("2015-03-04", "2017-03-04")
    X, y = training_data.features, training_data.target

    assert len(X) == len(y)
    assert len(training_data.dates) == len(y)
    assert training_data.dates.is_monotonic_increasing
    for forbidden_column in ["target", "date", "sets_p1", "sets_p2", "comment"]:
        assert forbidden_column not in X.columns


def test_training_exports_model_artifact(tmp_path):
    result = train_model(
        "2015-03-04", "2017-03-04", output_dir=tmp_path, export_joblib=True
    )

    assert result.exported_model_path is not None
    metadata_path = model_metadata_path(result.exported_model_path)
    card_path = model_card_path(result.exported_model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 2
    assert metadata["model"]["filename"] == result.exported_model_path.name
    assert metadata["model"]["kind"] == "elo_baseline"
    assert metadata["model"]["sha256"] == result.model_checksum
    assert metadata["feature_contract"]["columns"] == list(MODEL_FEATURE_COLUMNS)
    assert metadata["dataset"]["requested_window"] == {
        "start_date": "2015-03-04",
        "end_date": "2017-03-04",
    }
    assert metadata["selection"] == result.selection
    assert set(result.candidate_metrics) == {"constant", "elo", "lightgbm"}
    assert re.fullmatch(
        r"model_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}\.joblib",
        result.exported_model_path.name,
    )
    assert "Candidate comparison" in card_path.read_text(encoding="utf-8")


def test_export_model_fails_when_timestamped_artifact_already_exists(tmp_path):
    exported_at = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    exported_path = export_model({"model": "first"}, tmp_path, exported_at=exported_at)

    assert exported_path.name == "model_2026-01-01-00-00-00.joblib"
    with pytest.raises(FileExistsError, match="already exists"):
        export_model({"model": "second"}, tmp_path, exported_at=exported_at)


def test_select_model_features_uses_exact_ordered_allowlist():
    df = prepare_data().head(1).drop(columns=["comment"])
    df = df.assign(
        outcome="finished",
        odds=1.5,
        extra_comment="discard me",
    )
    shuffled = df.loc[:, list(reversed(df.columns))]

    features = select_model_features(shuffled)

    assert list(features.columns) == list(MODEL_FEATURE_COLUMNS)
    for discarded in [
        "target",
        "date",
        "atp",
        "location",
        "tournament",
        "outcome",
        "odds",
        "extra_comment",
    ]:
        assert discarded not in features.columns


@pytest.mark.parametrize("missing_column", MODEL_FEATURE_COLUMNS)
def test_select_model_features_reports_every_missing_required_column(missing_column):
    df = prepare_data().head(1).drop(columns=[missing_column])

    with pytest.raises(ValueError, match="missing required columns") as exc_info:
        select_model_features(df)

    assert missing_column in str(exc_info.value)


def test_login_uses_env_backed_demo_credentials(client: TestClient, monkeypatch):
    monkeypatch.setenv("ACEBET_SECRET_KEY", "test-signing-secret")
    monkeypatch.setenv("ACEBET_ENV", "test")
    monkeypatch.setenv("ACEBET_DEMO_USERNAME", "custom-user")
    monkeypatch.setenv("ACEBET_DEMO_PASSWORD", "custom-pass")
    get_settings.cache_clear()
    get_demo_users_db.cache_clear()

    response = client.post(
        "/token",
        data={"username": "custom-user", "password": "custom-pass"},
    )

    assert response.status_code == 200


class FixedEstimator:
    """Minimal fitted binary estimator with deterministic outputs."""

    feature_names_in_ = np.asarray(MODEL_FEATURE_COLUMNS)
    n_features_in_ = len(MODEL_FEATURE_COLUMNS)

    def __init__(
        self,
        probability: float = 0.312,
        predicted_class: int = 1,
        *,
        error: Exception | None = None,
        probabilities: object | None = None,
        classes: object | None = None,
    ) -> None:
        self.probability = probability
        self.predicted_class = predicted_class
        self.error = error
        self.probabilities = probabilities
        self.classes = classes

    def predict_proba(self, _features):
        if self.error is not None:
            raise self.error
        if self.probabilities is not None:
            return self.probabilities
        return np.asarray([[1 - self.probability, self.probability]])

    def predict(self, _features):
        if self.classes is not None:
            return self.classes
        return np.asarray([self.predicted_class])


def synthetic_player_stats(
    p1_as_of_date: str = "2026-01-10",
    p2_as_of_date: str = "2026-01-08",
) -> pd.DataFrame:
    """Return deterministic snapshots for API freshness tests."""
    return pd.DataFrame(
        [
            {
                "player": "Player One",
                "rank": 10,
                "elo": 1800.0,
                "as_of_date": pd.Timestamp(p1_as_of_date),
            },
            {
                "player": "Player Two",
                "rank": 20,
                "elo": 1700.0,
                "as_of_date": pd.Timestamp(p2_as_of_date),
            },
        ]
    ).set_index("player", drop=False)


def phase4_payload(**overrides: object) -> dict[str, object]:
    """Return a complete request for synthetic Phase 4 assets."""
    payload: dict[str, object] = {
        "p1_name": "Player One",
        "p2_name": "Player Two",
        "date": "2026-01-10",
        "series": "ATP500",
        "court": "Indoor",
        "surface": "Clay",
        "round": "Semifinals",
        "best_of": 3,
    }
    payload.update(overrides)
    return payload


def install_phase4_assets(
    client: TestClient,
    *,
    mode: str = "testing",
    estimator: object | None = None,
    player_stats: pd.DataFrame | None = None,
    identity: ModelIdentity | None = None,
) -> None:
    """Replace one mode's runtime assets with deterministic test doubles."""
    assets = PredictionAssets(
        model=estimator or FixedEstimator(),
        player_stats=synthetic_player_stats() if player_stats is None else player_stats,
        model_identity=identity
        or ModelIdentity(
            kind="elo_baseline",
            version=f"model_{mode}",
            sha256=("a" if mode == "testing" else "b") * 64,
        ),
    )
    setattr(client.app.state.runtime_assets, mode, assets)


@pytest.mark.parametrize(
    ("path", "mode", "predicted_class", "predicted_winner", "version", "sha"),
    [
        (
            "/predict/testing",
            "testing",
            1,
            "Player One",
            "model_testing",
            "a" * 64,
        ),
        (
            "/predict/production",
            "production",
            0,
            "Player Two",
            "model_production",
            "b" * 64,
        ),
    ],
)
def test_phase4_success_contract_uses_class_probability_identity_and_freshness(
    path,
    mode,
    predicted_class,
    predicted_winner,
    version,
    sha,
    client,
    auth_headers,
):
    install_phase4_assets(
        client,
        mode=mode,
        estimator=FixedEstimator(probability=0.312, predicted_class=predicted_class),
    )

    response = client.post(path, headers=auth_headers, json=phase4_payload())

    assert response.status_code == 200
    assert response.json() == {
        "predicted_winner": predicted_winner,
        "predicted_class": predicted_class,
        "p1_win_probability": 0.312,
        "prediction_mode": mode,
        "model": {
            "kind": "elo_baseline",
            "version": version,
            "sha256": sha,
        },
        "feature_freshness": {
            "p1_as_of_date": "2026-01-10",
            "p2_as_of_date": "2026-01-08",
            "p1_age_days": 0,
            "p2_age_days": 2,
        },
    }


def test_phase4_names_are_trimmed_but_remain_case_sensitive(client, auth_headers):
    install_phase4_assets(client)

    response = client.post(
        "/predict/testing",
        headers=auth_headers,
        json=phase4_payload(
            p1_name="  Player One  ",
            p2_name=" Player Two ",
        ),
    )
    wrong_case = client.post(
        "/predict/testing",
        headers=auth_headers,
        json=phase4_payload(p1_name="player one"),
    )

    assert response.status_code == 200
    assert response.json()["predicted_winner"] == "Player One"
    assert wrong_case.status_code == 422
    assert wrong_case.json()["detail"]["code"] == "unknown_players"


@pytest.mark.parametrize(
    ("field", "allowed_values"),
    list(PREDICTION_CONTEXT_ALLOWED_VALUES.items()),
)
def test_phase4_every_context_enum_value_is_accepted(field, allowed_values):
    for value in allowed_values:
        request = PredictionRequest.model_validate(phase4_payload(**{field: value}))
        assert getattr(request, field).value == value


@pytest.mark.parametrize(
    "overrides",
    [
        {"p1_name": "   "},
        {"date": "not-a-date"},
        {"best_of": "three"},
    ],
)
def test_phase4_ordinary_malformed_fields_keep_native_validation(
    overrides, client, auth_headers
):
    response = client.post(
        "/predict/testing",
        headers=auth_headers,
        json=phase4_payload(**overrides),
    )

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)


@pytest.mark.parametrize(
    "field",
    ["series", "court", "surface", "round", "best_of"],
)
def test_phase4_missing_context_fields_keep_native_validation(
    field, client, auth_headers
):
    payload = phase4_payload()
    del payload[field]

    response = client.post("/predict/testing", headers=auth_headers, json=payload)

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert any(error["loc"] == ["body", field] for error in response.json()["detail"])


@pytest.mark.parametrize("legacy_field", ["atp", "location", "tournament", "testing"])
def test_phase4_extra_and_legacy_fields_keep_native_validation(
    legacy_field, client, auth_headers
):
    response = client.post(
        "/predict/testing",
        headers=auth_headers,
        json=phase4_payload(**{legacy_field: "legacy"}),
    )

    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_phase4_equal_players_returns_exact_domain_error(client, auth_headers):
    response = client.post(
        "/predict/testing",
        headers=auth_headers,
        json=phase4_payload(
            p1_name="  Player One ",
            p2_name="Player One",
        ),
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "equal_players",
            "message": "Player names must be distinct.",
            "context": {"player": "Player One"},
        }
    }


def test_phase4_multiple_unsupported_values_return_request_order(client, auth_headers):
    response = client.post(
        "/predict/testing",
        headers=auth_headers,
        json=phase4_payload(series="Challenger", surface="Wood", best_of=7),
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "unsupported_context_values",
            "message": "One or more match context values are unsupported.",
            "context": {
                "invalid_fields": [
                    {
                        "field": "series",
                        "submitted_value": "Challenger",
                        "allowed_values": list(
                            PREDICTION_CONTEXT_ALLOWED_VALUES["series"]
                        ),
                    },
                    {
                        "field": "surface",
                        "submitted_value": "Wood",
                        "allowed_values": list(
                            PREDICTION_CONTEXT_ALLOWED_VALUES["surface"]
                        ),
                    },
                    {
                        "field": "best_of",
                        "submitted_value": 7,
                        "allowed_values": [3, 5],
                    },
                ]
            },
        }
    }


@pytest.mark.parametrize(
    ("p1_name", "p2_name", "missing"),
    [
        ("Missing One", "Player Two", ["Missing One"]),
        ("Missing One", "Missing Two", ["Missing One", "Missing Two"]),
    ],
)
def test_phase4_unknown_players_return_all_names_in_request_order(
    p1_name, p2_name, missing, client, auth_headers
):
    install_phase4_assets(client)

    response = client.post(
        "/predict/testing",
        headers=auth_headers,
        json=phase4_payload(p1_name=p1_name, p2_name=p2_name),
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "unknown_players",
            "message": "Player snapshots are unavailable for one or more requested players.",
            "context": {"players": missing},
        }
    }


@pytest.mark.parametrize(
    ("match_date", "p1_as_of", "p2_as_of", "expected_ages"),
    [
        ("2026-01-10", "2026-01-10", "2026-01-10", (0, 0)),
        ("2026-01-12", "2026-01-10", "2026-01-08", (2, 4)),
        ("2026-01-12", "2026-01-12", "2026-01-08", (0, 4)),
    ],
)
def test_phase4_all_non_negative_snapshot_ages_are_allowed(
    match_date, p1_as_of, p2_as_of, expected_ages, client, auth_headers
):
    install_phase4_assets(
        client,
        player_stats=synthetic_player_stats(p1_as_of, p2_as_of),
    )

    response = client.post(
        "/predict/testing",
        headers=auth_headers,
        json=phase4_payload(date=match_date),
    )

    assert response.status_code == 200
    freshness = response.json()["feature_freshness"]
    assert (freshness["p1_age_days"], freshness["p2_age_days"]) == expected_ages


@pytest.mark.parametrize(
    ("p1_as_of", "p2_as_of", "affected"),
    [
        (
            "2026-01-11",
            "2026-01-08",
            [
                {
                    "player": "Player One",
                    "role": "p1",
                    "as_of_date": "2026-01-11",
                    "age_days": -1,
                }
            ],
        ),
        (
            "2026-01-11",
            "2026-01-12",
            [
                {
                    "player": "Player One",
                    "role": "p1",
                    "as_of_date": "2026-01-11",
                    "age_days": -1,
                },
                {
                    "player": "Player Two",
                    "role": "p2",
                    "as_of_date": "2026-01-12",
                    "age_days": -2,
                },
            ],
        ),
    ],
)
def test_phase4_negative_snapshot_ages_return_every_affected_player(
    p1_as_of, p2_as_of, affected, client, auth_headers
):
    install_phase4_assets(
        client,
        player_stats=synthetic_player_stats(p1_as_of, p2_as_of),
    )

    response = client.post(
        "/predict/testing",
        headers=auth_headers,
        json=phase4_payload(),
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "negative_snapshot_age",
            "message": "Player snapshot dates must not be later than the match date.",
            "context": {
                "match_date": "2026-01-10",
                "affected_players": affected,
            },
        }
    }


@pytest.mark.parametrize("expected_error", [KeyError("feature"), ValueError("score")])
def test_phase4_expected_feature_and_scoring_errors_are_sanitized_422(
    expected_error, client, auth_headers
):
    install_phase4_assets(client, estimator=FixedEstimator(error=expected_error))

    response = client.post(
        "/predict/testing", headers=auth_headers, json=phase4_payload()
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "prediction_failed",
            "message": "Prediction could not be completed.",
            "context": {},
        }
    }
    assert "feature" not in response.text
    assert "score" not in response.text


@pytest.mark.parametrize(
    ("probabilities", "classes"),
    [
        ([[0.4, float("nan")]], [1]),
        ([[0.4, 0.6], [0.3, 0.7]], [1]),
        ([[0.4, 0.6]], [2]),
    ],
)
def test_phase4_malformed_estimator_outputs_are_prediction_failed(
    probabilities, classes, client, auth_headers
):
    install_phase4_assets(
        client,
        estimator=FixedEstimator(probabilities=probabilities, classes=classes),
    )

    response = client.post(
        "/predict/testing", headers=auth_headers, json=phase4_payload()
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "prediction_failed"
    assert response.json()["detail"]["context"] == {}


def test_phase4_unexpected_prediction_error_is_sanitized_500(client, auth_headers):
    install_phase4_assets(
        client, estimator=FixedEstimator(error=RuntimeError("internal secret"))
    )

    response = client.post(
        "/predict/testing", headers=auth_headers, json=phase4_payload()
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal prediction engine error."}
    assert "internal secret" not in response.text


def test_phase4_missing_runtime_assets_remains_503(client, auth_headers):
    install_phase4_assets(client, mode="production")
    client.app.state.runtime_assets.production = None

    response = client.post(
        "/predict/production", headers=auth_headers, json=phase4_payload()
    )

    assert response.status_code == 503


def test_phase4_asset_availability_precedes_player_lookup(client, auth_headers):
    install_phase4_assets(client, mode="production")
    client.app.state.runtime_assets.production = None

    response = client.post(
        "/predict/production",
        headers=auth_headers,
        json=phase4_payload(p1_name="Missing One"),
    )

    assert response.status_code == 503


def _openapi_property_names(value: object) -> set[str]:
    if isinstance(value, dict):
        names = set(value.get("properties", {}))
        for nested in value.values():
            names.update(_openapi_property_names(nested))
        return names
    if isinstance(value, list):
        names: set[str] = set()
        for nested in value:
            names.update(_openapi_property_names(nested))
        return names
    return set()


def test_phase4_openapi_exposes_only_final_prediction_contract():
    schema = app.openapi()
    request_schema = schema["components"]["schemas"]["PredictionRequest"]
    response_schema = schema["components"]["schemas"]["PredictionResponse"]

    assert request_schema["additionalProperties"] is False
    assert request_schema["required"] == [
        "p1_name",
        "p2_name",
        "date",
        "series",
        "court",
        "surface",
        "round",
        "best_of",
    ]
    assert set(request_schema["properties"]) == set(request_schema["required"])
    assert set(response_schema["required"]) == {
        "predicted_winner",
        "predicted_class",
        "p1_win_probability",
        "prediction_mode",
        "model",
        "feature_freshness",
    }
    assert _openapi_property_names(schema).isdisjoint(
        {"atp", "location", "tournament", "player_name", "prob", "class_"}
    )
    operation = schema["paths"]["/predict/production"]["post"]
    assert (
        operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/PredictionResponse"
    )
