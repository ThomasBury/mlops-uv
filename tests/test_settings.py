from __future__ import annotations

import io
import asyncio
from importlib.util import find_spec
import json
import logging
import os

import pytest
import structlog
from fastapi.testclient import TestClient
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from acebet.app.dependencies.auth import (
    get_demo_users_db,
    get_password_hash,
    verify_password,
)
from acebet.app.main import app, configure_logging, create_app
from acebet.app.observability import (
    build_span_exporter,
    configure_tracing,
    shutdown_tracing,
)
import acebet.app.observability as observability
from acebet.settings import AppSettings, get_jwt_signing_key, get_settings


def _tracing_packages_available() -> bool:
    try:
        return all(
            find_spec(module) is not None
            for module in (
                "opentelemetry.exporter.otlp.proto.http.trace_exporter",
                "opentelemetry.instrumentation.fastapi",
                "opentelemetry.sdk.trace",
            )
        )
    except ModuleNotFoundError:
        return False


requires_tracing = pytest.mark.skipif(
    not _tracing_packages_available(),
    reason="requires the optional tracing extra",
)


@pytest.fixture(autouse=True)
def clear_settings_caches(monkeypatch):
    for env_key in list(os.environ):
        if env_key.startswith("ACEBET_"):
            monkeypatch.delenv(env_key, raising=False)
    get_settings.cache_clear()
    get_demo_users_db.cache_clear()
    yield
    get_demo_users_db.cache_clear()
    get_settings.cache_clear()


def test_settings_defaults_load_without_env_vars(monkeypatch, tmp_path):
    settings = AppSettings(_env_file=None, _secrets_dir=tmp_path)

    assert settings.secret_key is None
    assert settings.allow_insecure_dev_secret is False
    assert settings.demo_username == "johndoe"
    assert settings.demo_password.get_secret_value() == "secret"
    assert settings.env == "development"
    assert settings.log_level == "INFO"
    assert settings.log_file is None
    assert settings.log_format == "pretty"
    assert settings.metrics_enabled is True
    assert settings.rate_limiting_enabled is True
    assert settings.token_rate_limit == "5/minute"
    assert settings.prediction_rate_limit == "30/minute"
    assert settings.tracing_enabled is False
    assert settings.trace_exporter == "console"
    assert settings.otlp_endpoint is None
    assert settings.service_name == "acebet-api"
    assert settings.model_dir is None
    assert settings.data_dir is None


def test_settings_read_values_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "ACEBET_SECRET_KEY=env-file-secret",
                "ACEBET_DEMO_USERNAME=env-file-user",
                "ACEBET_LOG_LEVEL=warning",
                "ACEBET_LOG_FORMAT=json",
                "ACEBET_RATE_LIMITING_ENABLED=false",
                "ACEBET_TOKEN_RATE_LIMIT=2/minute",
                "ACEBET_PREDICTION_RATE_LIMIT=10/minute",
                "ACEBET_TRACING_ENABLED=true",
            ]
        ),
        encoding="utf-8",
    )

    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    settings = AppSettings(_env_file=env_file, _secrets_dir=secrets_dir)

    assert settings.secret_key is not None
    assert settings.secret_key.get_secret_value() == "env-file-secret"
    assert settings.demo_username == "env-file-user"
    assert settings.log_level == "WARNING"
    assert settings.log_format == "json"
    assert settings.rate_limiting_enabled is False
    assert settings.token_rate_limit == "2/minute"
    assert settings.prediction_rate_limit == "10/minute"
    assert settings.tracing_enabled is True


def test_environment_variables_override_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ACEBET_DEMO_USERNAME=env-file-user\n", encoding="utf-8")
    monkeypatch.setenv("ACEBET_DEMO_USERNAME", "real-env-user")
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()

    settings = AppSettings(_env_file=env_file, _secrets_dir=secrets_dir)

    assert settings.demo_username == "real-env-user"


def test_secret_files_are_used_when_env_and_env_file_do_not_provide_value(tmp_path):
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    (secrets_dir / "ACEBET_SECRET_KEY").write_text("from-secret-file", encoding="utf-8")

    settings = AppSettings(_env_file=None, _secrets_dir=secrets_dir)

    assert settings.secret_key is not None
    assert settings.secret_key.get_secret_value() == "from-secret-file"


def test_get_settings_honors_acebet_secrets_dir_override(tmp_path, monkeypatch):
    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    (secrets_dir / "ACEBET_SECRET_KEY").write_text("override-secret", encoding="utf-8")
    monkeypatch.setenv("ACEBET_SECRETS_DIR", str(secrets_dir))

    settings = get_settings()

    assert settings.secrets_dir == str(secrets_dir)
    assert settings.secret_key is not None
    assert settings.secret_key.get_secret_value() == "override-secret"


@pytest.mark.parametrize("process_override", [False, True])
def test_get_settings_resolves_dotenv_secret_directory_with_env_precedence(
    tmp_path, monkeypatch, process_override
):
    monkeypatch.chdir(tmp_path)
    dotenv_dir = tmp_path / "dotenv-secrets"
    dotenv_dir.mkdir()
    (dotenv_dir / "ACEBET_SECRET_KEY").write_text("dotenv-directory-secret")
    (tmp_path / ".env").write_text(f"ACEBET_SECRETS_DIR={dotenv_dir}\n")
    expected_secret = "dotenv-directory-secret"
    expected_dir = dotenv_dir
    if process_override:
        expected_dir = tmp_path / "process-secrets"
        expected_dir.mkdir()
        expected_secret = "process-directory-secret"
        (expected_dir / "ACEBET_SECRET_KEY").write_text(expected_secret)
        monkeypatch.setenv("ACEBET_SECRETS_DIR", str(expected_dir))

    settings = get_settings()

    assert settings.secrets_dir == str(expected_dir)
    assert get_jwt_signing_key() == expected_secret


def test_development_fallback_secret_starts_successfully(monkeypatch):
    monkeypatch.setenv("ACEBET_ENV", "development")
    monkeypatch.setenv("ACEBET_ALLOW_INSECURE_DEV_SECRET", "true")

    with TestClient(app):
        resolved_key = get_jwt_signing_key()

    assert resolved_key


def test_missing_secret_fails_when_insecure_fallback_is_disabled(monkeypatch):
    monkeypatch.setenv("ACEBET_ENV", "development")
    monkeypatch.setenv("ACEBET_ALLOW_INSECURE_DEV_SECRET", "false")

    with pytest.raises(
        RuntimeError,
        match="ACEBET_SECRET_KEY is required unless ACEBET_ENV=development",
    ):
        with TestClient(app):
            pass


def test_non_development_env_cannot_use_bundled_fallback(monkeypatch):
    monkeypatch.setenv("ACEBET_ENV", "production")
    monkeypatch.setenv("ACEBET_ALLOW_INSECURE_DEV_SECRET", "true")
    monkeypatch.setenv("ACEBET_DEMO_PASSWORD", "production-password")

    with pytest.raises(
        RuntimeError,
        match="ACEBET_SECRET_KEY is required unless ACEBET_ENV=development",
    ):
        with TestClient(app):
            pass


def test_configure_logging_uses_stream_handler_when_log_file_is_unset(tmp_path):
    settings = AppSettings(
        _env_file=None,
        _secrets_dir=tmp_path,
        log_file=None,
    )

    configure_logging(settings)

    root_logger = logging.getLogger()
    assert any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    )


def test_configure_logging_uses_file_handler_when_log_file_is_set(tmp_path):
    log_path = tmp_path / "acebet.log"
    settings = AppSettings(
        _env_file=None,
        _secrets_dir=tmp_path,
        log_file=str(log_path),
    )

    configure_logging(settings)

    root_logger = logging.getLogger()
    assert any(
        isinstance(handler, logging.FileHandler) for handler in root_logger.handlers
    )


def test_configure_logging_emits_json_with_request_context(tmp_path):
    stream = io.StringIO()
    settings = AppSettings(
        _env_file=None,
        _secrets_dir=tmp_path,
        log_format="json",
    )

    configure_logging(settings, stream=stream)
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id="req-123")
    structlog.get_logger("acebet.test").info("json_log_test")
    structlog.contextvars.clear_contextvars()

    payload = json.loads(stream.getvalue().strip())
    assert payload["event"] == "json_log_test"
    assert payload["level"] == "info"
    assert payload["request_id"] == "req-123"


@pytest.mark.tracing
@requires_tracing
def test_build_span_exporter_uses_console_by_default(tmp_path):
    settings = AppSettings(_env_file=None, _secrets_dir=tmp_path)

    exporter = build_span_exporter(settings)

    assert exporter.__class__.__name__ == "ConsoleSpanExporter"


def test_build_span_exporter_requires_otlp_endpoint(tmp_path):
    settings = AppSettings(
        _env_file=None,
        _secrets_dir=tmp_path,
        trace_exporter="otlp",
        otlp_endpoint=None,
    )

    with pytest.raises(RuntimeError, match="ACEBET_OTLP_ENDPOINT is required"):
        build_span_exporter(settings)


def test_configure_tracing_validates_otlp_before_optional_imports(
    monkeypatch, tmp_path
):
    settings = AppSettings(
        _env_file=None,
        _secrets_dir=tmp_path,
        tracing_enabled=True,
        trace_exporter="otlp",
        otlp_endpoint=None,
    )

    def fail_if_imported():
        raise AssertionError("optional tracing imports must not be attempted")

    monkeypatch.setattr(observability, "_load_tracing_modules", fail_if_imported)

    with pytest.raises(RuntimeError, match="ACEBET_OTLP_ENDPOINT is required"):
        configure_tracing(FastAPI(), settings)


def test_configure_tracing_returns_none_when_disabled(tmp_path):
    settings = AppSettings(_env_file=None, _secrets_dir=tmp_path, tracing_enabled=False)

    assert configure_tracing(FastAPI(), settings) is None


@pytest.mark.tracing
@requires_tracing
def test_configure_tracing_loads_and_releases_optional_runtime(tmp_path):
    application = FastAPI()
    settings = AppSettings(
        _env_file=None,
        _secrets_dir=tmp_path,
        tracing_enabled=True,
    )

    provider = configure_tracing(application, settings)

    assert provider is not None
    assert application.state.tracer is not None
    shutdown_tracing(application)
    assert application.state.tracer_provider is None


@pytest.mark.tracing
@requires_tracing
def test_lifespan_tracing_exports_http_spans_and_restarts(monkeypatch):
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.trace import SpanKind

    monkeypatch.setenv("ACEBET_TRACING_ENABLED", "true")
    monkeypatch.setenv("ACEBET_SECRET_KEY", "tracing-regression-secret")
    application = create_app()

    async def exercise_lifespan():
        messages = []

        async def receive():
            if not messages:
                return {"type": "lifespan.startup"}
            async with AsyncClient(
                transport=ASGITransport(app=application), base_url="http://test"
            ) as client:
                assert (await client.get("/openapi.json")).status_code == 200
                assert (await client.get("/metrics")).status_code == 200
            return {"type": "lifespan.shutdown"}

        async def send(message):
            messages.append(message)

        await application({"type": "lifespan", "state": {}}, receive, send)
        assert [message["type"] for message in messages] == [
            "lifespan.startup.complete",
            "lifespan.shutdown.complete",
        ]

    for _ in range(2):
        exporter = InMemorySpanExporter()
        monkeypatch.setattr(
            observability, "build_span_exporter", lambda *args: exporter
        )
        asyncio.run(asyncio.wait_for(exercise_lifespan(), timeout=10))
        server_spans = [
            span
            for span in exporter.get_finished_spans()
            if span.kind == SpanKind.SERVER
        ]
        assert len(server_spans) == 1
        assert server_spans[0].name == "GET /openapi.json"
        assert application.state.tracer_provider is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("env", "staging"),
        ("log_format", "xml"),
        ("trace_exporter", "zipkin"),
        ("jwt_algorithm", "none"),
        ("access_token_expire_minutes", 0),
        ("access_token_expire_minutes", -1),
        ("service_name", "   "),
    ],
)
def test_settings_reject_invalid_finite_values_and_constraints(tmp_path, field, value):
    with pytest.raises(ValueError):
        AppSettings(
            _env_file=None,
            _secrets_dir=tmp_path,
            **{field: value},
        )


@pytest.mark.parametrize("env", ["development", "test"])
def test_non_production_environments_accept_tutorial_password(tmp_path, env):
    settings = AppSettings(_env_file=None, _secrets_dir=tmp_path, env=env)

    assert settings.demo_password.get_secret_value() == "secret"


@pytest.mark.parametrize("password", ["secret", "too-short"])
def test_production_rejects_default_or_short_demo_password(tmp_path, password):
    with pytest.raises(ValueError, match="ACEBET_DEMO_PASSWORD"):
        AppSettings(
            _env_file=None,
            _secrets_dir=tmp_path,
            env="production",
            demo_password=password,
        )


def test_production_accepts_explicit_long_demo_password(tmp_path):
    settings = AppSettings(
        _env_file=None,
        _secrets_dir=tmp_path,
        env="production",
        demo_password="a-production-password",
    )

    assert settings.env == "production"


@pytest.mark.parametrize("env", ["development", "test", "production"])
@pytest.mark.parametrize("password", ["a" * 73, "é" * 37])
def test_settings_reject_passwords_over_bcrypt_byte_limit(tmp_path, env, password):
    with pytest.raises(ValueError, match="at most 72 UTF-8 bytes"):
        AppSettings(
            _env_file=None, _secrets_dir=tmp_path, env=env, demo_password=password
        )


@pytest.mark.parametrize("password", ["a" * 72, "é" * 36])
def test_settings_accept_passwords_at_bcrypt_byte_limit(tmp_path, password):
    settings = AppSettings(
        _env_file=None, _secrets_dir=tmp_path, env="production", demo_password=password
    )

    assert settings.demo_password.get_secret_value() == password


def test_authentication_does_not_truncate_overlong_login_password():
    password = "é" * 36
    hashed = get_password_hash(password)

    assert verify_password(password, hashed)
    assert not verify_password(password + "extra", hashed)


@pytest.mark.parametrize(
    "method",
    [
        "GET",
        "HEAD",
        "POST",
        "PUT",
        "DELETE",
        "CONNECT",
        "OPTIONS",
        "TRACE",
        "PATCH",
        "CUSTOM",
        "BREW",
        "user-method-123",
    ],
)
@pytest.mark.parametrize("path", ["/predict/production", "__unmatched__"])
def test_http_method_metrics_have_bounded_shared_labels(method, path):
    from prometheus_client import REGISTRY

    label = method if method in observability.STANDARD_HTTP_METHODS else "OTHER"
    counter_labels = {"method": label, "path": path, "status_code": "405"}
    histogram_labels = {"method": label, "path": path}
    before_count = (
        REGISTRY.get_sample_value("acebet_http_requests_total", counter_labels) or 0
    )
    before_duration_count = (
        REGISTRY.get_sample_value(
            "acebet_http_request_duration_seconds_count", histogram_labels
        )
        or 0
    )
    observability.record_http_metrics(method, path, 405, 0.01)
    assert (
        REGISTRY.get_sample_value("acebet_http_requests_total", counter_labels)
        == before_count + 1
    )
    assert (
        REGISTRY.get_sample_value(
            "acebet_http_request_duration_seconds_count", histogram_labels
        )
        == before_duration_count + 1
    )
    if label == "OTHER":
        metrics = observability.render_metrics()[0].decode()
        assert f'method="{method}"' not in metrics
