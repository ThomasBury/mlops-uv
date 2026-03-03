"""Application configuration loaded and validated from environment variables."""

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEV_ENVIRONMENTS = {"dev", "development", "local", "test", "testing"}


class AppSettings(BaseSettings):
    """Application settings sourced from environment variables."""

    model_config = SettingsConfigDict(extra="ignore")

    acebet_env: str = Field(default="development", alias="ACEBET_ENV")
    acebet_secret_key: str | None = Field(default=None, alias="ACEBET_SECRET_KEY")
    acebet_jwt_algorithm: str = Field(default="HS256", alias="ACEBET_JWT_ALGORITHM")
    acebet_access_token_expire_minutes: int = Field(
        default=30,
        alias="ACEBET_ACCESS_TOKEN_EXPIRE_MINUTES",
    )
    log_level: str = Field(default="INFO", alias="ACEBET_LOG_LEVEL")
    log_file: str = Field(default="acebet.log", alias="ACEBET_LOG_FILE")
    default_rate_limit: str = Field(
        default="60/minute", alias="ACEBET_DEFAULT_RATE_LIMIT"
    )
    login_rate_limit: str = Field(default="10/minute", alias="ACEBET_LOGIN_RATE_LIMIT")

    @field_validator("acebet_access_token_expire_minutes")
    @classmethod
    def validate_expiry_minutes(cls, value: int) -> int:
        """Ensure token expiration minutes is a positive integer."""
        if value <= 0:
            raise ValueError(
                "ACEBET_ACCESS_TOKEN_EXPIRE_MINUTES must be greater than 0."
            )
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Ensure log level matches one of the standard ``logging`` levels."""
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        normalized = value.strip().upper()
        if normalized not in allowed:
            raise ValueError("ACEBET_LOG_LEVEL must be a valid Python logging level.")
        return normalized

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

    def redacted(self) -> dict[str, str | int]:
        """Return non-sensitive configuration values for debug logging."""
        return {
            "acebet_env": self.acebet_env,
            "acebet_jwt_algorithm": self.acebet_jwt_algorithm,
            "acebet_access_token_expire_minutes": (
                self.acebet_access_token_expire_minutes
            ),
            "log_level": self.log_level,
            "log_file": self.log_file,
            "default_rate_limit": self.default_rate_limit,
            "login_rate_limit": self.login_rate_limit,
            "acebet_secret_key": (
                "***redacted***" if self.acebet_secret_key else "<dev-default>"
            ),
        }


settings = AppSettings()

ACEBET_SECRET_KEY = settings.acebet_secret_key or "acebet-dev-insecure-secret-key"
ACEBET_JWT_ALGORITHM = settings.acebet_jwt_algorithm
ACEBET_ACCESS_TOKEN_EXPIRE_MINUTES = settings.acebet_access_token_expire_minutes


def validate_config() -> None:
    """Validate startup configuration and raise clear runtime errors when invalid."""
    try:
        AppSettings()
    except ValidationError as exc:
        raise RuntimeError(f"Invalid application configuration: {exc}") from exc
