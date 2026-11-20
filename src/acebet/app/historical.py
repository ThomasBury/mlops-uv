"""Serve the frozen tutorial forecast through FastAPI and Uvicorn."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import Counter, Gauge, Histogram

from acebet.app.observability import (
    configure_logging,
    configure_tracing,
    get_request_logger,
    prediction_span,
    render_metrics,
    shutdown_tracing,
    user_logging_middleware,
)
from acebet.historical_forecast import (
    ForecastRequest,
    ForecastResponse,
    InvalidReplay,
    forecast_id,
    load_forecaster,
)
from acebet.settings import ObservabilitySettings
from acebet.train.historical_evaluation import CALIBRATION_BINS

FORECASTS_IN_PROGRESS = Gauge(
    "acebet_forecasts_in_progress", "Historical forecast handlers currently running."
)
FORECAST_SURFACES = Counter(
    "acebet_forecast_surface_total",
    "Successful forecasts by input surface.",
    ["surface"],
)
FORECAST_ELO_GAP = Histogram(
    "acebet_forecast_absolute_elo_gap",
    "Absolute Elo gap submitted to the model for successful forecasts.",
    buckets=(0, 25, 50, 100, 200, 400, 800),
)
FORECAST_PROBABILITY = Histogram(
    "acebet_forecast_p1_win_probability",
    "Win probability returned for the first player in the request.",
    buckets=CALIBRATION_BINS[1:],
)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = ObservabilitySettings()
        configure_logging(settings)
        try:
            configure_tracing(app, settings)
            app.state.metrics_enabled = settings.metrics_enabled
            try:
                app.state.forecaster = load_forecaster(
                    Path(os.environ["ACEBET_FORECAST_RESULTS"]),
                    Path(os.environ["ACEBET_FORECAST_BUNDLE"]),
                )
            except Exception as exc:
                app.state.forecaster = None
                get_request_logger().error(
                    "forecast_assets_invalid",
                    error_type=type(exc).__name__,
                    reason=str(exc),
                )
            yield
        finally:
            app.state.forecaster = None
            shutdown_tracing(app)

    app = FastAPI(
        title="AceBet historical forecast",
        lifespan=lifespan,
        telemetry={
            "tracing": False,
            "metrics": False,
            "logs": False,
            "operation_spans": False,
            "auto_configure": False,
        },
    )
    app.middleware("http")(user_logging_middleware)

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/readyz")
    async def readiness(request: Request) -> dict[str, str]:
        if getattr(request.app.state, "forecaster", None) is None:
            raise HTTPException(status_code=503, detail="Forecast assets unavailable")
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    def metrics(request: Request) -> Response:
        if not request.app.state.metrics_enabled:
            raise HTTPException(status_code=404, detail="Metrics endpoint disabled")
        payload, content_type = render_metrics()
        return Response(content=payload, media_type=content_type)

    @app.post("/forecast")
    def forecast(payload: ForecastRequest, request: Request) -> ForecastResponse:
        forecaster = getattr(request.app.state, "forecaster", None)
        if forecaster is None:
            raise HTTPException(status_code=503, detail="Forecast assets unavailable")
        with FORECASTS_IN_PROGRESS.track_inprogress():
            try:
                result = forecaster.predict(
                    payload, span=partial(prediction_span, request)
                )
            except InvalidReplay as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            FORECAST_SURFACES.labels(surface=result.surface).inc()
            FORECAST_ELO_GAP.observe(abs(result.elo_gap))
            FORECAST_PROBABILITY.observe(result.p1_win_probability)
            get_request_logger().info(
                "forecast_served",
                forecast_id=forecast_id(result),
                forecast=result.model_dump(mode="json"),
            )
            return result

    return app
