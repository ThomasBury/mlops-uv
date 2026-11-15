"""Application settings and secret resolution for AceBet."""

from __future__ import annotations

import os

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    PositiveInt,
    SecretStr,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, DotEnvSettingsSource, SettingsConfigDict

INSECURE_TUTORIAL_SECRET = (
    "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
)
DEFAULT_SECRET_DIRS = ("/var/run", "/run/secrets")
TUTORIAL_DEMO_PASSWORD = "secret"

Environment = Literal["development", "test", "production"]
JwtAlgorithm = Literal["HS256", "HS384", "HS512"]
LogFormat = Literal["pretty", "json"]
TraceExporter = Literal["console", "otlp"]
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AppSettings(BaseSettings):
    """Central configuration for the tutorial service."""

    secret_key: SecretStr | None = None
    secrets_dir: str | None = None
    allow_insecure_dev_secret: bool = False
    jwt_algorithm: JwtAlgorithm = "HS256"
    access_token_expire_minutes: PositiveInt = 30
    demo_username: str = "johndoe"
    demo_password: SecretStr = SecretStr(TUTORIAL_DEMO_PASSWORD)
    env: Environment = "development"
    log_level: str = "INFO"
    log_file: str | None = None
    log_format: LogFormat = "pretty"
    metrics_enabled: bool = True
    rate_limiting_enabled: bool = True
    token_rate_limit: str = "5/minute"
    prediction_rate_limit: str = "30/minute"
    tracing_enabled: bool = False
    trace_exporter: TraceExporter = "console"
    otlp_endpoint: str | None = None
    service_name: NonEmptyString = "acebet-api"
    model_dir: Path | None = None
    data_dir: Path | None = None

    model_config = SettingsConfigDict(
        env_prefix="ACEBET_",
        env_file=".env",
        env_file_encoding="utf-8",
        secrets_dir=("/var/run", "/run/secrets"),
        extra="ignore",
    )

    @field_validator(
        "secret_key",
        "secrets_dir",
        "log_file",
        "otlp_endpoint",
        "model_dir",
        "data_dir",
        mode="before",
    )
    @classmethod
    def blank_strings_to_none(cls, value: object) -> object:
        """Treat blank settings as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("env", mode="before")
    @classmethod
    def normalize_env(cls, value: object) -> object:
        """Normalize the environment name."""
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("jwt_algorithm", "log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        """Normalize the configured log level."""
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("log_format", "trace_exporter", mode="before")
    @classmethod
    def normalize_observability_strings(cls, value: object) -> object:
        """Normalize observability string settings."""
        if isinstance(value, str):
            return value.strip().lower() if value.strip() else value.strip()
        return value

    @field_validator("demo_password")
    @classmethod
    def validate_demo_password_byte_length(cls, value: SecretStr) -> SecretStr:
        """Reject passwords that bcrypt would silently truncate."""
        if len(value.get_secret_value().encode("utf-8")) > 72:
            raise ValueError("ACEBET_DEMO_PASSWORD must be at most 72 UTF-8 bytes.")
        return value

    @model_validator(mode="after")
    def validate_production_demo_password(self) -> AppSettings:
        """Reject tutorial credentials when the service runs in production."""
        if self.env != "production":
            return self
        password = self.demo_password.get_secret_value()
        if password == TUTORIAL_DEMO_PASSWORD or len(password) < 12:
            raise ValueError(
                "ACEBET_DEMO_PASSWORD must be explicitly set to at least 12 "
                "characters in production and cannot use the tutorial default."
            )
        return self


@lru_cache
def get_settings() -> AppSettings:
    """Load settings once per process with runtime secret-dir override support."""
    configured_secrets_dir = os.getenv("ACEBET_SECRETS_DIR")
    if configured_secrets_dir is None:
        configured_secrets_dir = DotEnvSettingsSource(AppSettings)().get("secrets_dir")
    if configured_secrets_dir and configured_secrets_dir.strip():
        secrets_dir: str | tuple[str, ...] = configured_secrets_dir
    else:
        secrets_dir = tuple(path for path in DEFAULT_SECRET_DIRS if os.path.isdir(path))
    return AppSettings(_secrets_dir=secrets_dir)


def get_jwt_signing_key() -> str:
    """Resolve the JWT signing key or raise a clear startup error."""
    settings = get_settings()
    if settings.secret_key is not None:
        return settings.secret_key.get_secret_value()
    if settings.env == "development" and settings.allow_insecure_dev_secret:
        return INSECURE_TUTORIAL_SECRET
    raise RuntimeError(
        "ACEBET_SECRET_KEY is required unless ACEBET_ENV=development and "
        "ACEBET_ALLOW_INSECURE_DEV_SECRET=true are both set."
    )
