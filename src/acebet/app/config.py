"""Application configuration loaded and validated from environment variables."""

from typing import Any

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_ENVIRONMENTS = {"dev", "development", "local", "test", "testing"}


class AppSettings(BaseSettings):
    """Application settings sourced from environment variables."""

    model_config = SettingsConfigDict(extra="ignore")

    acebet_env: str = Field(default="development", alias="ACEBET_ENV")
    acebet_secret_key: str | None = Field(default=None, alias="ACEBET_SECRET_KEY")
    acebet_jwt_algorithm: str = Field(default="HS256", alias="ACEBET_JWT_ALGORITHM")
    acebet_log_level: str = Field(default="INFO", alias="ACEBET_LOG_LEVEL")
    acebet_log_file: str | None = Field(default=None, alias="ACEBET_LOG_FILE")
    acebet_default_rate_limit: str = Field(
        default="100/minute", alias="ACEBET_DEFAULT_RATE_LIMIT"
    )
    acebet_login_rate_limit: str = Field(
        default="5/minute", alias="ACEBET_LOGIN_RATE_LIMIT"
    )
    acebet_access_token_expire_minutes: int = Field(
        default=30,
        alias="ACEBET_ACCESS_TOKEN_EXPIRE_MINUTES",
    )

    @field_validator("acebet_access_token_expire_minutes")
    @classmethod
    def validate_expiry_minutes(cls, value: int) -> int:
        """Ensure token expiration minutes is a positive integer."""
        if value <= 0:
            raise ValueError(
                "ACEBET_ACCESS_TOKEN_EXPIRE_MINUTES must be greater than 0."
            )
        return value

    @model_validator(mode="after")
    def validate_secret_key_in_non_dev(self) -> "AppSettings":
        """Require a secret key outside development-like environments."""
        env_name = self.acebet_env.strip().lower()
        if env_name not in DEV_ENVIRONMENTS and not self.acebet_secret_key:
            raise ValueError(
                "Missing required environment variable ACEBET_SECRET_KEY in "
                "non-development environments."
            )
        return self

    def redacted(self) -> dict[str, Any]:
        """Return a safe settings representation for structured logging."""
        payload = self.model_dump()
        if payload.get("acebet_secret_key"):
            payload["acebet_secret_key"] = "***REDACTED***"
        return payload


settings = AppSettings()


def validate_config() -> None:
    """Validate startup configuration and raise clear runtime errors when invalid."""
    try:
        AppSettings()
    except ValidationError as exc:
        raise RuntimeError(f"Invalid application configuration: {exc}") from exc
