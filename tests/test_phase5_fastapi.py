"""Phase 5 regressions for the modular FastAPI application."""

import asyncio
import inspect
import re
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from acebet.app import main
from acebet.app.routers import prediction as prediction_router
from acebet.app.dependencies.auth import (
    get_current_active_user,
    get_current_user,
    get_demo_users_db,
)
from acebet.app.main import app, create_app
from acebet.app.observability import shutdown_tracing
from acebet.app.runtime import PredictionAssets
from acebet.features import MODEL_FEATURE_COLUMNS
from acebet.model_artifacts import read_json
from acebet.settings import AppSettings, get_settings

pytestmark = pytest.mark.api


@pytest.fixture
def phase5_settings(monkeypatch):
    monkeypatch.setenv("ACEBET_SECRET_KEY", "phase5-test-signing-secret")
    monkeypatch.setenv("ACEBET_ENV", "test")
    monkeypatch.setenv("ACEBET_RATE_LIMITING_ENABLED", "false")
    get_settings.cache_clear()
    get_demo_users_db.cache_clear()
    yield
    get_demo_users_db.cache_clear()
    get_settings.cache_clear()


def test_create_app_returns_independent_app_owned_state():
    first = create_app()
    second = create_app()

    assert app is not first
    assert first is not second
    assert first.state.limiter is not second.state.limiter
    assert first.state.runtime_assets is not second.state.runtime_assets


def test_lifespan_loads_and_clears_both_runtime_modes(phase5_settings):
    application = create_app()

    assert application.state.runtime_assets.production is None
    assert application.state.runtime_assets.testing is None
    with TestClient(application):
        assert application.state.runtime_assets.production is not None
        assert application.state.runtime_assets.testing is not None
        promotion = read_json(main.PACKAGE_DATA_DIR / "promoted_model.json")
        identity = application.state.runtime_assets.production.model_identity
        assert identity.version == promotion["model_filename"].removesuffix(".joblib")
        assert identity.kind == promotion["model_kind"]
        assert identity.sha256 == promotion["model_sha256"]

    assert application.state.runtime_assets.production is None
    assert application.state.runtime_assets.testing is None


def test_production_lifespan_loads_only_production(monkeypatch):
    monkeypatch.setenv("ACEBET_SECRET_KEY", "phase5-test-signing-secret")
    monkeypatch.setenv("ACEBET_ENV", "production")
    monkeypatch.setenv("ACEBET_DEMO_PASSWORD", "phase5-production-password")
    monkeypatch.setenv("ACEBET_RATE_LIMITING_ENABLED", "false")
    get_settings.cache_clear()
    application = create_app()
    try:
        with TestClient(application):
            assert application.state.runtime_assets.production is not None
            assert application.state.runtime_assets.testing is None
    finally:
        get_settings.cache_clear()

    assert application.state.runtime_assets.production is None
    assert application.state.runtime_assets.testing is None


def test_partial_startup_failure_clears_loaded_assets(monkeypatch, phase5_settings):
    real_loader = main.load_prediction_assets
    calls = 0

    def fail_testing_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("testing load failed")
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(main, "load_prediction_assets", fail_testing_load)
    application = create_app()

    with pytest.raises(RuntimeError, match="testing load failed"):
        with TestClient(application):
            pass

    assert application.state.runtime_assets.production is None
    assert application.state.runtime_assets.testing is None


def test_tracing_flushes_before_shutdown_and_clears_state():
    order: list[str] = []

    class FakeProvider:
        def force_flush(self):
            order.append("force_flush")

        def shutdown(self):
            order.append("shutdown")

    class FakeInstrumentor:
        @staticmethod
        def uninstrument_app(_app):
            order.append("uninstrument")

    application = create_app()
    application.state.tracer_provider = FakeProvider()
    application.state.tracer = object()
    application.state.tracing_instrumentor = FakeInstrumentor
    application.state.tracing_instrumented = True

    shutdown_tracing(application)

    assert order == ["uninstrument", "force_flush", "shutdown"]
    assert application.state.tracer_provider is None
    assert application.state.tracer is None
    assert application.state.tracing_instrumentor is None
    assert application.state.tracing_instrumented is False


class NoEventLoopEstimator:
    feature_names_in_ = np.asarray(MODEL_FEATURE_COLUMNS)
    n_features_in_ = len(MODEL_FEATURE_COLUMNS)

    """Estimator that proves scoring runs in FastAPI's worker threadpool."""

    def _assert_no_event_loop(self):
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()

    def predict_proba(self, _features):
        self._assert_no_event_loop()
        return np.asarray([[0.25, 0.75]])

    def predict(self, _features):
        self._assert_no_event_loop()
        return np.asarray([1])


def test_prediction_scoring_and_auth_dependencies_are_synchronous(phase5_settings):
    application = create_app()
    prediction_endpoints = {
        route.name: route.endpoint
        for route in prediction_router.create_prediction_router(
            application.state.limiter
        ).routes
        if route.name.startswith("predict_match_outcome_")
    }

    assert not inspect.iscoroutinefunction(get_current_user)
    assert not inspect.iscoroutinefunction(get_current_active_user)
    assert set(prediction_endpoints) == {
        "predict_match_outcome_production",
        "predict_match_outcome_testing",
    }
    assert all(
        not inspect.iscoroutinefunction(endpoint)
        for endpoint in prediction_endpoints.values()
    )

    with TestClient(application) as client:
        token = client.post(
            "/token", data={"username": "johndoe", "password": "secret"}
        ).json()["access_token"]
        loaded = application.state.runtime_assets.testing
        application.state.runtime_assets.testing = PredictionAssets(
            model=NoEventLoopEstimator(),
            player_stats=loaded.player_stats,
            model_identity=loaded.model_identity,
        )
        response = client.post(
            "/predict/testing",
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

    assert response.status_code == 200


def test_local_rate_limit_handler_returns_exact_body(monkeypatch):
    monkeypatch.setenv("ACEBET_SECRET_KEY", "phase5-test-signing-secret")
    monkeypatch.setenv("ACEBET_ENV", "test")
    monkeypatch.setenv("ACEBET_RATE_LIMITING_ENABLED", "true")
    monkeypatch.setenv("ACEBET_TOKEN_RATE_LIMIT", "1/minute")
    get_settings.cache_clear()
    get_demo_users_db.cache_clear()
    application = create_app()
    try:
        with TestClient(application) as client:
            first = client.post(
                "/token", data={"username": "johndoe", "password": "secret"}
            )
            limited = client.post(
                "/token", data={"username": "johndoe", "password": "secret"}
            )
    finally:
        get_demo_users_db.cache_clear()
        get_settings.cache_clear()

    assert first.status_code == 200
    assert limited.status_code == 429
    assert limited.json() == {"error": "Rate limit exceeded: 1 per 1 minute"}


def test_request_context_cleanup_runs_when_metrics_processing_fails(
    monkeypatch, phase5_settings
):
    cleanup_calls = 0
    real_clear = main.clear_request_context

    def tracked_clear():
        nonlocal cleanup_calls
        cleanup_calls += 1
        real_clear()

    def fail_metrics(*_args, **_kwargs):
        raise RuntimeError("metrics failed")

    monkeypatch.setattr(main, "clear_request_context", tracked_clear)
    monkeypatch.setattr(main, "record_http_metrics", fail_metrics)
    application = create_app()

    with TestClient(application) as client:
        with pytest.raises(RuntimeError, match="metrics failed"):
            client.get("/healthz")

    assert cleanup_calls == 1


def _unmatched_404_count(metrics: str) -> float:
    match = re.search(
        r'^acebet_http_requests_total\{method="GET",path="__unmatched__",status_code="404"\} ([0-9.eE+-]+)$',
        metrics,
        re.MULTILINE,
    )
    return 0.0 if match is None else float(match.group(1))


def test_arbitrary_404s_share_only_the_unmatched_metric_label(phase5_settings):
    application = create_app()
    with TestClient(application) as client:
        before = _unmatched_404_count(client.get("/metrics").text)
        assert client.get("/missing-phase5-alpha").status_code == 404
        assert client.get("/missing-phase5-beta").status_code == 404
        metrics = client.get("/metrics").text

    assert _unmatched_404_count(metrics) == before + 2
    assert 'path="/missing-phase5-alpha"' not in metrics
    assert 'path="/missing-phase5-beta"' not in metrics


def test_users_me_has_a_canonical_no_slash_path(phase5_settings):
    application = create_app()
    with TestClient(application) as client:
        token = client.post(
            "/token", data={"username": "johndoe", "password": "secret"}
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        canonical = client.get("/users/me", headers=headers)
        trailing_slash = client.get(
            "/users/me/", headers=headers, follow_redirects=False
        )

    assert canonical.status_code == 200
    assert trailing_slash.status_code == 404
    assert "location" not in trailing_slash.headers


def test_openapi_paths_and_operational_schemas_are_precise():
    schema = app.openapi()

    assert set(schema["paths"]) == {
        "/",
        "/healthz",
        "/readyz",
        "/token",
        "/users/me",
        "/predict/production",
        "/predict/testing",
    }
    assert schema["paths"]["/"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/WelcomeResponse"}
    assert schema["paths"]["/healthz"]["get"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/HealthResponse"}
    ready_responses = schema["paths"]["/readyz"]["get"]["responses"]
    assert ready_responses["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ReadyResponse"
    }
    assert ready_responses["503"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/NotReadyResponse"
    }


def _phase6_prediction_payload():
    return {
        "p1_name": "Fognini F.",
        "p2_name": "Jarry N.",
        "date": "2018-03-04",
        "series": "ATP250",
        "court": "Outdoor",
        "surface": "Hard",
        "round": "1st Round",
        "best_of": 3,
    }


@pytest.mark.parametrize(
    ("stage", "error", "status_code"),
    [
        ("snapshot", ValueError("invalid stats"), 422),
        ("snapshot", RuntimeError("snapshot engine"), 500),
        ("freshness", ValueError("invalid freshness"), 422),
        ("freshness", RuntimeError("freshness engine"), 500),
    ],
)
def test_prediction_stages_map_expected_and_unexpected_errors(
    monkeypatch, phase5_settings, stage, error, status_code
):
    target = (
        "resolve_player_snapshots"
        if stage == "snapshot"
        else "calculate_feature_freshness"
    )
    monkeypatch.setattr(prediction_router, target, Mock(side_effect=error))
    application = create_app()
    with TestClient(application, raise_server_exceptions=False) as client:
        token = client.post(
            "/token", data={"username": "johndoe", "password": "secret"}
        ).json()["access_token"]
        response = client.post(
            "/predict/testing",
            headers={"Authorization": f"Bearer {token}"},
            json=_phase6_prediction_payload(),
        )

    assert response.status_code == status_code
    if status_code == 422:
        assert response.json()["detail"]["code"] == "prediction_failed"
    else:
        assert response.json()["detail"] == "Internal prediction engine error."


def test_asset_resolution_uses_packaged_defaults(tmp_path, monkeypatch):
    package_data = tmp_path / "package"
    package_data.mkdir()
    (package_data / "latest_player_stats.feather").touch()
    (package_data / "promoted_model.json").touch()
    monkeypatch.setattr(main, "PACKAGE_DATA_DIR", package_data)
    settings = AppSettings(_env_file=None, _secrets_dir=tmp_path)

    assert main.resolve_prediction_assets("production", settings) == (
        package_data / "latest_player_stats.feather",
        package_data,
    )


def test_configured_asset_directories_take_independent_precedence(
    tmp_path, monkeypatch
):
    package_data = tmp_path / "package"
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    for directory in (package_data, data_dir, model_dir):
        directory.mkdir()
    (package_data / "latest_player_stats.feather").touch()
    (package_data / "promoted_model.json").touch()
    (data_dir / "latest_player_stats.feather").touch()
    (model_dir / "promoted_model.json").touch()
    monkeypatch.setattr(main, "PACKAGE_DATA_DIR", package_data)

    data_override = AppSettings(
        _env_file=None, _secrets_dir=tmp_path, data_dir=data_dir
    )
    assert main.resolve_prediction_assets("production", data_override) == (
        data_dir / "latest_player_stats.feather",
        package_data,
    )

    both_overrides = AppSettings(
        _env_file=None,
        _secrets_dir=tmp_path,
        data_dir=data_dir,
        model_dir=model_dir,
    )
    assert main.resolve_prediction_assets("production", both_overrides) == (
        data_dir / "latest_player_stats.feather",
        model_dir,
    )


def test_invalid_configured_asset_paths_never_fall_back(tmp_path, monkeypatch):
    package_data = tmp_path / "package"
    package_data.mkdir()
    (package_data / "latest_player_stats.feather").touch()
    (package_data / "promoted_model.json").touch()
    monkeypatch.setattr(main, "PACKAGE_DATA_DIR", package_data)

    missing_data = tmp_path / "missing-data"
    settings = AppSettings(_env_file=None, _secrets_dir=tmp_path, data_dir=missing_data)
    with pytest.raises(FileNotFoundError, match=str(missing_data)):
        main.resolve_prediction_assets("production", settings)

    missing_model = tmp_path / "missing-model"
    settings = AppSettings(
        _env_file=None, _secrets_dir=tmp_path, model_dir=missing_model
    )
    with pytest.raises(FileNotFoundError, match=str(missing_model)):
        main.resolve_prediction_assets("production", settings)


def test_testing_asset_resolution_is_isolated_from_production_overrides(
    tmp_path, monkeypatch
):
    package_data = tmp_path / "package"
    external = tmp_path / "external"
    package_data.mkdir()
    external.mkdir()
    (package_data / "latest_player_stats_sample.feather").touch()
    (package_data / "promoted_model.json").touch()
    (external / "latest_player_stats.feather").touch()
    (external / "promoted_model.json").touch()
    monkeypatch.setattr(main, "PACKAGE_DATA_DIR", package_data)
    settings = AppSettings(
        _env_file=None,
        _secrets_dir=tmp_path,
        data_dir=external,
        model_dir=external,
    )

    assert main.resolve_prediction_assets("testing", settings) == (
        package_data / "latest_player_stats_sample.feather",
        package_data,
    )


def test_prediction_feature_contract_requires_two_players():
    player_stats = pd.DataFrame(
        [{"player": "only", "rank": 1, "elo": 2000, "as_of_date": "2025-01-01"}]
    )
    with pytest.raises(ValueError, match="at least two players"):
        main.assert_prediction_feature_contract(Mock(), player_stats)


def test_lifespan_tolerates_shutdown_observability_failures(
    monkeypatch, phase5_settings
):
    events = []

    class ShutdownLogger:
        def info(self, event, **_context):
            if event == "api_shutdown":
                events.append(event)
                raise RuntimeError("logger closed")

        def exception(self, event, **_context):
            events.append(event)

    def fail_shutdown(_application):
        raise RuntimeError("trace shutdown failed")

    monkeypatch.setattr(main, "get_request_logger", lambda: ShutdownLogger())
    monkeypatch.setattr(main, "shutdown_tracing", fail_shutdown)
    application = create_app()

    with TestClient(application):
        pass

    assert events == ["tracing_shutdown_failed", "api_shutdown"]


def test_logging_middleware_records_and_reraises_application_errors(
    monkeypatch, phase5_settings
):
    recorded = []
    monkeypatch.setattr(
        main,
        "record_http_metrics",
        lambda method, path, status, duration: recorded.append(
            (method, path, status, duration)
        ),
    )
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/explodes",
            "raw_path": b"/explodes",
            "query_string": b"",
            "headers": [],
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 1),
        }
    )

    async def fail(_request):
        raise RuntimeError("application failed")

    with pytest.raises(RuntimeError, match="application failed"):
        asyncio.run(main.user_logging_middleware(request, fail))

    assert recorded[0][0:3] == ("GET", "__unmatched__", 500)


def test_native_telemetry_stays_disabled_with_environment_autoconfiguration(
    monkeypatch,
):
    monkeypatch.setenv("FASTAPI_OTEL_AUTO_CONFIGURE", "true")
    application = create_app()
    for setting in ("tracing", "metrics", "logs", "operation_spans", "auto_configure"):
        assert application._telemetry[setting] is False


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("player", None, "nonblank strings"),
        ("player", "  ", "nonblank strings"),
        ("player", 42, "nonblank strings"),
        ("player", "Alpha", "unique"),
        ("as_of_date", None, "parseable and non-null"),
        ("as_of_date", "invalid", "parseable and non-null"),
        ("rank", float("inf"), "finite numeric"),
        ("rank", float("nan"), "finite numeric"),
        ("rank", None, "finite numeric"),
        ("elo", None, "finite numeric"),
        ("rank", 0, "positive integers"),
        ("rank", -1, "positive integers"),
        ("rank", 1.5, "positive integers"),
        ("elo", float("-inf"), "finite numeric"),
        ("elo", "invalid", "finite numeric"),
        ("missing", None, "missing columns"),
    ],
)
def test_malformed_external_player_state_fails_startup_and_clears_assets(
    tmp_path, monkeypatch, phase5_settings, column, value, message
):
    stats = pd.DataFrame(
        {
            "player": ["Alpha", "Beta", "Gamma"],
            "as_of_date": ["2025-01-01"] * 3,
            "rank": [1.0, 2.0, 3.0],
            "elo": [2000.0, 1900.0, 1800.0],
        }
    )
    if column == "missing":
        stats = stats.drop(columns="elo")
    elif column == "player" and value == 42:
        stats["player"] = [1, 2, 42]
    elif column == "elo" and value == "invalid":
        stats["elo"] = ["2000", "1900", "invalid"]
    else:
        if column in ("rank", "elo") and value is None:
            stats[column] = stats[column].astype("Float64")
        stats.loc[2, column] = value
    stats.to_feather(tmp_path / "latest_player_stats.feather")
    monkeypatch.setenv("ACEBET_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    application = create_app()
    with pytest.raises(ValueError, match=message):
        with TestClient(application):
            pass
    assert application.state.runtime_assets.production is None
    assert application.state.runtime_assets.testing is None


def test_startup_feature_check_uses_loaded_snapshot_values(monkeypatch):
    stats = pd.DataFrame(
        {
            "player": ["Alpha", "Beta"],
            "rank": [31, 47],
            "elo": [1701.5, 1602.5],
            "as_of_date": ["2025-01-01"] * 2,
        }
    )
    build = Mock(wraps=main.build_match_feature_row)
    monkeypatch.setattr(main, "build_match_feature_row", build)
    monkeypatch.setattr(main, "align_features_to_model", Mock())
    main.assert_prediction_feature_contract(Mock(), stats)
    assert build.call_args.kwargs["p1_snapshot"] == {"rank": 31.0, "elo": 1701.5}
    assert build.call_args.kwargs["p2_snapshot"] == {"rank": 47.0, "elo": 1602.5}


def test_custom_request_methods_share_series_on_matched_and_unmatched_paths(
    phase5_settings,
):
    from prometheus_client import REGISTRY

    methods = ["CUSTOM-ONE", "CUSTOM-TWO", "BREW"]
    application = create_app()
    application.add_api_route("/method-probe", lambda: {"ok": True}, methods=methods)
    with TestClient(application) as client:
        for path, metric_path, status in (
            ("/method-probe", "/method-probe", 200),
            ("/unknown-method-probe", "__unmatched__", 404),
        ):
            counter_labels = {
                "method": "OTHER",
                "path": metric_path,
                "status_code": str(status),
            }
            duration_labels = {"method": "OTHER", "path": metric_path}
            before = (
                REGISTRY.get_sample_value("acebet_http_requests_total", counter_labels)
                or 0
            )
            duration_before = (
                REGISTRY.get_sample_value(
                    "acebet_http_request_duration_seconds_count", duration_labels
                )
                or 0
            )
            for method in methods:
                assert client.request(method, path).status_code == status
            assert REGISTRY.get_sample_value(
                "acebet_http_requests_total", counter_labels
            ) == before + len(methods)
            assert REGISTRY.get_sample_value(
                "acebet_http_request_duration_seconds_count", duration_labels
            ) == duration_before + len(methods)
        metrics = client.get("/metrics").text
        assert all(f'method="{method}"' not in metrics for method in methods)


@pytest.mark.parametrize("fitted", [False, True])
def test_startup_requires_a_fitted_classifier_in_a_checksum_valid_bundle(
    tmp_path, monkeypatch, phase5_settings, fitted
):
    from sklearn.exceptions import NotFittedError

    from acebet.features import assert_model_artifact_compatible
    from acebet.model_artifacts import (
        candidates_with_selection_details,
        compute_selection,
        model_metadata_path,
        sha256_file,
        write_json,
    )
    from acebet.train.promote import promote_model
    from acebet.train.train import (
        build_model,
        export_model,
        get_lgb_params,
        prepare_data_for_training_clf,
        write_model_card,
    )

    data = prepare_data_for_training_clf("2015-03-04", "2017-03-04")
    model = build_model(get_lgb_params())
    if fitted:
        model.fit(data.features, data.target)
    else:
        model.named_steps["preprocessor"].fit(data.features, data.target)
    assert_model_artifact_compatible(model)
    path = export_model(model, tmp_path)
    metadata = read_json(
        main.PACKAGE_DATA_DIR / "model_2026-08-02-12-59-07.metadata.json"
    )
    metrics = {
        name: dict(metadata["candidates"][name]["metrics"])
        for name in ("constant", "elo", "lightgbm")
    }
    # Synthetic selection evidence keeps the fixture valid up to the fitted-state check.
    metrics["lightgbm"]["log_loss"] = metrics["elo"]["log_loss"] - 0.01
    metadata["candidates"] = candidates_with_selection_details(
        metrics, metadata["candidates"]["training_prevalence"], get_lgb_params()
    )
    metadata["selection"] = compute_selection(metadata["candidates"])
    metadata["model"].update(
        filename=path.name, kind="lightgbm", sha256=sha256_file(path)
    )
    write_json(model_metadata_path(path), metadata)
    write_model_card(path, metadata)
    promote_model(model_path=path)
    monkeypatch.setenv("ACEBET_MODEL_DIR", str(tmp_path))
    get_settings.cache_clear()
    application = create_app()
    if fitted:
        with TestClient(application) as client:
            assert client.get("/readyz").status_code == 200
            assert (
                application.state.runtime_assets.production.model_identity.kind
                == "lightgbm"
            )
    else:
        with pytest.raises(NotFittedError, match="not fitted"):
            with TestClient(application):
                pytest.fail("Startup accepted an unfitted classifier")
    assert application.state.runtime_assets.production is None
    assert application.state.runtime_assets.testing is None
