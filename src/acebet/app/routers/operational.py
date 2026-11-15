"""Operational and service-discovery routes."""

from typing import Literal

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel
from slowapi import Limiter

from acebet.app.dependencies.runtime import RuntimeAssetsDep
from acebet.app.runtime import RuntimeAssets
from acebet.settings import AppSettings, get_settings

WELCOME_MESSAGE = "Welcome to the AceBet API!"


class WelcomeResponse(BaseModel):
    """Stable service welcome response."""

    message: Literal["Welcome to the AceBet API!"]


class HealthResponse(BaseModel):
    """Liveness response."""

    status: Literal["ok"]


class ReadyResponse(BaseModel):
    """Successful readiness response."""

    status: Literal["ready"]


class NotReadyResponse(BaseModel):
    """Unavailable readiness response."""

    status: Literal["not_ready"]


def testing_predictions_enabled(settings: AppSettings | None = None) -> bool:
    """Return whether the sample/testing prediction route is enabled."""
    resolved_settings = settings or get_settings()
    return resolved_settings.env in {"development", "test"}


def is_app_ready(runtime_assets: RuntimeAssets) -> bool:
    """Return whether all environment-required serving assets are present."""
    return runtime_assets.is_ready(require_testing=testing_predictions_enabled())


def create_operational_router(limiter: Limiter) -> APIRouter:
    """Build operational routes bound to one application's limiter."""
    router = APIRouter()

    @router.get("/", response_model=WelcomeResponse)
    @limiter.exempt
    def home() -> WelcomeResponse:
        """Return the stable API welcome message."""
        return WelcomeResponse(message=WELCOME_MESSAGE)

    @router.get("/healthz", response_model=HealthResponse)
    @limiter.exempt
    def healthz() -> HealthResponse:
        """Return a minimal liveness response."""
        return HealthResponse(status="ok")

    @router.get(
        "/readyz",
        response_model=ReadyResponse,
        responses={
            status.HTTP_503_SERVICE_UNAVAILABLE: {
                "model": NotReadyResponse,
                "description": "Required prediction assets are not loaded.",
            }
        },
    )
    @limiter.exempt
    def readyz(runtime_assets: RuntimeAssetsDep) -> ReadyResponse | Response:
        """Return readiness based on startup-loaded serving assets."""
        if is_app_ready(runtime_assets):
            return ReadyResponse(status="ready")
        return Response(
            content='{"status":"not_ready"}',
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            media_type="application/json",
        )

    @router.get("/metrics", include_in_schema=False, response_model=None)
    @limiter.exempt
    def metrics() -> Response:
        """Return Prometheus metrics for the tutorial API."""
        from acebet.app.observability import render_metrics

        settings = get_settings()
        if not settings.metrics_enabled:
            raise HTTPException(status_code=404, detail="Metrics endpoint disabled")
        payload, content_type = render_metrics()
        return Response(content=payload, media_type=content_type)

    return router
