"""Application configuration with explicit source precedence."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment sources.

    Precedence order is:
    1. Process environment variables.
    2. Local ``.env`` values (loaded only when keys are not already set).
    3. In-code defaults for non-sensitive values.
    """

    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    default_rate_limit: str = "12/minute"
    login_rate_limit: str = "5/minute"
    log_level: str = "DEBUG"
    log_file: str = "info.log"

    def redacted(self) -> dict[str, str | int]:
        """Return configuration values safe for debug logging."""
        return {
            "algorithm": self.algorithm,
            "access_token_expire_minutes": self.access_token_expire_minutes,
            "default_rate_limit": self.default_rate_limit,
            "login_rate_limit": self.login_rate_limit,
            "log_level": self.log_level,
            "log_file": self.log_file,
        }


def load_settings() -> Settings:
    """Load application settings with env > .env > default precedence."""
    load_dotenv(override=False)

    secret_key = os.getenv("ACEBET_SECRET_KEY")
    if not secret_key:
        raise ValueError(
            "Missing ACEBET_SECRET_KEY. Set it in the process environment "
            "or a local .env file."
        )

    return Settings(
        secret_key=secret_key,
        algorithm=os.getenv("ACEBET_ALGORITHM", "HS256"),
        access_token_expire_minutes=int(
            os.getenv("ACEBET_ACCESS_TOKEN_EXPIRE_MINUTES", "30")
        ),
        default_rate_limit=os.getenv("ACEBET_DEFAULT_RATE_LIMIT", "12/minute"),
        login_rate_limit=os.getenv("ACEBET_LOGIN_RATE_LIMIT", "5/minute"),
        log_level=os.getenv("ACEBET_LOG_LEVEL", "DEBUG"),
        log_file=os.getenv("ACEBET_LOG_FILE", "info.log"),
    )


settings = load_settings()
