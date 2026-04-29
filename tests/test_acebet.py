import pytest
from fastapi.testclient import TestClient

from acebet.app.main import app
from acebet.app.dependencies.auth import get_password_hash, verify_password
from acebet.dataprep.dataprep import prepare_data
from acebet.features import (
    build_latest_player_stats,
    build_match_feature_row,
    select_model_features,
)
from acebet.train.train import prepare_data_for_training_clf, train_model


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def access_token(client: TestClient) -> str:
    response = client.post(
        "/token",
        data={"username": "johndoe", "password": "secret"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture
def auth_headers(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


def test_login_for_access_token(client: TestClient):
    response = client.post(
        "/token",
        data={"username": "johndoe", "password": "secret"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_bcrypt_helpers_support_seeded_and_new_hashes():
    seeded_hash = "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"

    assert verify_password("secret", seeded_hash)

    hashed_password = get_password_hash("another-secret")
    assert hashed_password.startswith("$2b$")
    assert verify_password("another-secret", hashed_password)
    assert not verify_password("wrong-password", hashed_password)


def test_read_users_me(client: TestClient, auth_headers: dict[str, str]):
    response = client.get("/users/me/", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["username"] == "johndoe"


def test_read_own_items(client: TestClient, auth_headers: dict[str, str]):
    response = client.get("/users/me/items/", headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["owner"] == "johndoe"


def test_predict_match_outcome(client: TestClient, auth_headers: dict[str, str]):
    response = client.post(
        "/predict/",
        headers=auth_headers,
        json={
            "p1_name": "Fognini F.",
            "p2_name": "Jarry N.",
            "date": "2018-03-04",
            "testing": True,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "player_name" in data
    assert "prob" in data
    assert "class_" in data
    assert 0.0 <= data["prob"] <= 100.0
    assert data["class_"] in [0, 1]


def test_predict_match_outcome_with_production_data(
    client: TestClient, auth_headers: dict[str, str]
):
    response = client.post(
        "/predict/",
        headers=auth_headers,
        json={
            "p1_name": "Fognini F.",
            "p2_name": "Jarry N.",
            "date": "2018-03-04",
            "testing": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "player_name" in data
    assert "prob" in data
    assert "class_" in data
    assert 0.0 <= data["prob"] <= 100.0
    assert data["class_"] in [0, 1]


def test_predict_match_outcome_for_future_match_uses_player_state(
    client: TestClient, auth_headers: dict[str, str]
):
    response = client.post(
        "/predict/",
        headers=auth_headers,
        json={
            "p1_name": "Fognini F.",
            "p2_name": "Jarry N.",
            "date": "2030-01-01",
            "testing": True,
            "surface": "Clay",
            "round": "Final",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["player_name"] == "Fognini F."
    assert 0.0 <= data["prob"] <= 100.0
    assert data["class_"] in [0, 1]


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
    features = build_match_feature_row(
        p1_name="Fognini F.",
        p2_name="Jarry N.",
        date="2030-01-01",
        stats_context=stats,
        match_context={"surface": "Clay", "round": "Final"},
    )

    assert features.loc[0, "p1"] == "Fognini F."
    assert features.loc[0, "p2"] == "Jarry N."
    assert features.loc[0, "surface"] == "Clay"
    assert features.loc[0, "round"] == "Final"
    assert features.loc[0, "year"] == 2030
    assert features.loc[0, "best_ranked"] in ["p1", "p2"]
    assert 0.0 <= features.loc[0, "proba_elo"] <= 1.0


def test_feature_contract_excludes_post_match_columns():
    X, y = prepare_data_for_training_clf("2015-03-04", "2017-03-04")

    assert len(X) == len(y)
    for forbidden_column in ["target", "date", "sets_p1", "sets_p2", "comment"]:
        assert forbidden_column not in X.columns


def test_training_exports_model_artifact(tmp_path):
    result = train_model(
        "2015-03-04",
        "2017-03-04",
        output_dir=tmp_path,
        export_joblib=True,
    )

    assert result.exported_model_path is not None
    assert result.exported_model_path.exists()
    assert "accuracy" in result.metrics
    assert "log_loss" in result.metrics
    assert 0.0 <= result.metrics["accuracy"] <= 1.0


def test_select_model_features_tolerates_missing_optional_drop_columns():
    df = prepare_data().head(1).drop(columns=["comment"])
    features = select_model_features(df)

    assert "comment" not in features.columns
    assert "target" not in features.columns
