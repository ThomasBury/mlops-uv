"""AceBet FastAPI application composition and lifespan management."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Any, cast
from fastapi import FastAPI, Request, Response
import pandas as pd
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from acebet.app.dependencies.data_models import (
    PREDICTION_CONTEXT_FIELDS,
)
from acebet.app.dependencies.predict_winner import (
    LoadedModelBundle,
    load_model_bundle,
    load_player_stats,
    resolve_player_snapshots,
)
from acebet.app.observability import (
    configure_logging,
    configure_tracing,
    get_request_logger,
    shutdown_tracing,
    user_logging_middleware,
)
from acebet.app.routers.authentication import create_authentication_router
from acebet.app.routers.operational import (
    create_operational_router,
    testing_predictions_enabled,
)
from acebet.app.routers.prediction import PREDICTION_PATHS, create_prediction_router
from acebet.app.runtime import PredictionAssets, PredictionMode, RuntimeAssets
from acebet.features import (
    PredictionEstimator,
    align_features_to_model,
    build_match_feature_row,
)
from acebet.model_artifacts import promotion_record_path, resolve_model_artifact_path
from acebet.settings import AppSettings, get_jwt_signing_key, get_settings

PACKAGE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def resolve_prediction_assets(
    prediction_mode: PredictionMode, settings: AppSettings
) -> tuple[Path, Path]:
    """Locate the player-state and promoted-model files for one runtime mode."""
    if prediction_mode == "testing":
        stats_file = PACKAGE_DATA_DIR / "latest_player_stats_sample.feather"
        model_dir = PACKAGE_DATA_DIR
    else:
        stats_file = (settings.data_dir or PACKAGE_DATA_DIR) / (
            "latest_player_stats.feather"
        )
        model_dir = settings.model_dir or PACKAGE_DATA_DIR

    if not stats_file.is_file():
        raise FileNotFoundError(
            f"Required player stats file not found for prediction_mode="
            f"{prediction_mode!r}: '{stats_file}'."
        )
    promotion_path = promotion_record_path(model_dir)
    if not promotion_path.is_file():
        raise FileNotFoundError(
            f"Required promoted model pointer not found for prediction_mode="
            f"{prediction_mode!r}: '{promotion_path}'."
        )
    return stats_file, model_dir


def assert_prediction_feature_contract(
    model: PredictionEstimator, player_stats: pd.DataFrame
) -> None:
    """Fail fast when loaded serving assets do not match the model schema."""
    players = list(dict.fromkeys(player_stats["player"].dropna().tolist()))
    if len(players) < 2:
        raise ValueError("Player stats table must contain at least two players.")

    p1_snapshot, p2_snapshot = resolve_player_snapshots(
        player_stats, players[0], players[1]
    )
    online_features = build_match_feature_row(
        p1_name=players[0],
        p2_name=players[1],
        date="2030-01-01",
        p1_snapshot=p1_snapshot.feature_values(),
        p2_snapshot=p2_snapshot.feature_values(),
        series="ATP250",
        court="Outdoor",
        surface="Hard",
        round="1st Round",
        best_of=3,
    )
    align_features_to_model(model, online_features)


def load_prediction_assets(
    prediction_mode: PredictionMode,
    model_cache: dict[Path, LoadedModelBundle],
    settings: AppSettings,
) -> PredictionAssets:
    """Load one complete set of estimator, player state, and model identity."""
    stats_file, model_path = resolve_prediction_assets(prediction_mode, settings)
    cache_key = resolve_model_artifact_path(model_path).resolve()
    bundle = model_cache.get(cache_key)
    if bundle is None:
        bundle = load_model_bundle(cache_key)
        model_cache[cache_key] = bundle
    expected_stats_sha256 = None
    if prediction_mode == "production":
        expected_stats_sha256 = bundle.player_stats_sha256
        if settings.player_stats_sha256 is not None:
            if expected_stats_sha256 not in (None, settings.player_stats_sha256):
                raise ValueError(
                    "ACEBET_PLAYER_STATS_SHA256 disagrees with the promoted bundle."
                )
            expected_stats_sha256 = settings.player_stats_sha256
        if expected_stats_sha256 is None and (settings.data_dir or settings.model_dir):
            raise ValueError(
                "Overridden assets require a player stats digest. Export a bound bundle "
                "or set ACEBET_PLAYER_STATS_SHA256 for an older bundle."
            )
    player_stats = load_player_stats(stats_file, expected_sha256=expected_stats_sha256)
    assert_prediction_feature_contract(bundle.estimator, player_stats)
    return PredictionAssets(
        model=bundle.estimator,
        player_stats=player_stats,
        model_identity=bundle.identity,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load serving assets at startup and release all app-owned state on exit."""
    settings = get_settings()
    configure_logging(settings)
    runtime_assets: RuntimeAssets = app.state.runtime_assets
    try:
        configure_tracing(app, settings)
        app.state.limiter.enabled = settings.rate_limiting_enabled
        get_jwt_signing_key()

        model_cache: dict[Path, LoadedModelBundle] = {}
        try:
            runtime_assets.production = load_prediction_assets(
                prediction_mode="production",
                model_cache=model_cache,
                settings=settings,
            )
            if testing_predictions_enabled(settings):
                runtime_assets.testing = load_prediction_assets(
                    prediction_mode="testing",
                    model_cache=model_cache,
                    settings=settings,
                )
        except Exception:
            get_request_logger().exception("startup_assets_load_failed")
            raise

        get_request_logger().info(
            "startup_assets_loaded",
            testing_predictions_enabled=testing_predictions_enabled(settings),
        )
        yield
    finally:
        runtime_assets.clear()
        try:
            shutdown_tracing(app)
        except Exception:
            get_request_logger().exception("tracing_shutdown_failed")
        try:
            get_request_logger().info("api_shutdown")
        except Exception:
            pass


async def prediction_request_validation_handler(
    request: Request, exc: RequestValidationError
) -> Response:
    """Map only prediction-specific custom validation tags to domain errors."""
    errors = exc.errors()
    error_types = {error["type"] for error in errors}
    if request.url.path in PREDICTION_PATHS and error_types == {"equal_players"}:
        context = errors[0].get("ctx") or {}
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "equal_players",
                    "message": "Player names must be distinct.",
                    "context": {"player": context["player"]},
                }
            },
        )
    if request.url.path in PREDICTION_PATHS and error_types == {
        "unsupported_context_value"
    }:
        field_order = {
            field: index for index, field in enumerate(PREDICTION_CONTEXT_FIELDS)
        }
        invalid_fields = sorted(
            (error.get("ctx") or {} for error in errors),
            key=lambda context: field_order[context["field"]],
        )
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "unsupported_context_values",
                    "message": "One or more match context values are unsupported.",
                    "context": {"invalid_fields": invalid_fields},
                }
            },
        )
    return await request_validation_exception_handler(request, exc)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Return the stable local JSON response for an exceeded rate limit."""
    return JSONResponse(
        {"error": f"Rate limit exceeded: {exc.detail}"}, status_code=429
    )


def create_app() -> FastAPI:
    """Create an independently stateful AceBet FastAPI application."""
    limiter = Limiter(key_func=get_remote_address)
    application = FastAPI(
        title="AceBet Match Predictor",
        description="MLOps Tutorial API for Tennis Match Winner Prediction",
        lifespan=lifespan,
        redirect_slashes=False,
        telemetry={
            "tracing": False,
            "metrics": False,
            "logs": False,
            "operation_spans": False,
            "auto_configure": False,
        },
    )
    application.state.limiter = limiter
    application.state.runtime_assets = RuntimeAssets()
    application.state.tracer_provider = None
    application.state.tracer = None
    application.state.tracing_instrumentor = None
    application.state.tracing_instrumented = False

    application.add_exception_handler(
        RateLimitExceeded, cast(Any, rate_limit_exceeded_handler)
    )
    application.add_exception_handler(
        RequestValidationError, cast(Any, prediction_request_validation_handler)
    )
    application.middleware("http")(user_logging_middleware)
    application.include_router(create_operational_router(limiter))
    application.include_router(create_authentication_router(limiter))
    application.include_router(create_prediction_router(limiter))
    return application


app = create_app()
