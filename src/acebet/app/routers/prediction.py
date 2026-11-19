"""Authenticated synchronous prediction routes."""

from typing import Any, NoReturn
import time

from fastapi import APIRouter, HTTPException, Request, Response
from slowapi import Limiter

from acebet.app.dependencies.api import CurrentActiveUserDep
from acebet.app.dependencies.auth import get_prediction_rate_limit_key
from acebet.app.dependencies.data_models import (
    PredictionDomainErrorResponse,
    PredictionRequest,
    PredictionResponse,
)
from acebet.app.dependencies.predict_winner import (
    NegativeSnapshotAgeError,
    StaleSnapshotsError,
    UnknownPlayerSnapshotsError,
    calculate_feature_freshness,
    make_online_prediction,
    resolve_player_snapshots,
)
from acebet.app.dependencies.runtime import RuntimeAssetsDep
from acebet.app.observability import (
    get_request_logger,
    prediction_span,
    record_prediction_metrics,
    record_sample_data_prediction,
    request_duration_seconds,
)
from acebet.app.routers.operational import testing_predictions_enabled
from acebet.app.runtime import PredictionAssets, PredictionMode, RuntimeAssets
from acebet.settings import get_settings

PREDICTION_PATHS = {"/predict/production", "/predict/testing"}
PREDICTION_DOMAIN_RESPONSES: dict[int | str, dict[str, Any]] = {
    422: {
        "model": PredictionDomainErrorResponse,
        "description": "Prediction-domain validation or scoring failure.",
    }
}


def prediction_rate_limit() -> str:
    """Return the configured shared prediction route rate limit."""
    return get_settings().prediction_rate_limit


def _get_prediction_assets(
    runtime_assets: RuntimeAssets, prediction_mode: PredictionMode
) -> PredictionAssets:
    """Return the preloaded assets for the requested prediction mode."""
    assets = runtime_assets.lookup(prediction_mode)
    if assets is None:
        raise HTTPException(
            status_code=503,
            detail="ML components not initialized. Please check server logs.",
        )
    return assets


def _raise_prediction_domain_error(
    *, code: str, message: str, context: dict[str, object]
) -> NoReturn:
    """Raise one stable structured prediction-domain response."""
    raise HTTPException(
        status_code=422,
        detail={"code": code, "message": message, "context": context},
    )


def _record_prediction_failure(
    prediction_mode: PredictionMode,
    prediction_start: float,
    error_type: str,
) -> None:
    """Record an expected or unexpected prediction failure."""
    record_prediction_metrics(
        mode=prediction_mode,
        duration_seconds=request_duration_seconds(prediction_start),
        error_type=error_type,
    )


def _predict_match_outcome(
    request: Request,
    runtime_assets: RuntimeAssets,
    prediction_request: PredictionRequest,
    prediction_mode: PredictionMode,
) -> PredictionResponse:
    """Run an authenticated prediction against one explicit asset mode."""
    p1_name = prediction_request.p1_name
    p2_name = prediction_request.p2_name
    match_date = prediction_request.date
    date_string = match_date.isoformat()
    sample_data = prediction_mode == "testing"
    assets = _get_prediction_assets(runtime_assets, prediction_mode)

    prediction_start = time.perf_counter()
    with prediction_span(request, "acebet.predict_match_outcome"):
        try:
            p1_snapshot, p2_snapshot = resolve_player_snapshots(
                assets.player_stats, p1_name, p2_name
            )
            feature_freshness = calculate_feature_freshness(
                match_date,
                p1_snapshot,
                p2_snapshot,
                max_age_days=get_settings().max_snapshot_age_days,
            )
            probability, predicted_class = make_online_prediction(
                model=assets.model,
                p1_name=p1_name,
                p2_name=p2_name,
                date=date_string,
                p1_snapshot=p1_snapshot,
                p2_snapshot=p2_snapshot,
                series=prediction_request.series.value,
                court=prediction_request.court.value,
                surface=prediction_request.surface.value,
                round=prediction_request.round.value,
                best_of=prediction_request.best_of.value,
            )
        except UnknownPlayerSnapshotsError as exc:
            _record_prediction_failure(
                prediction_mode, prediction_start, "unknown_players"
            )
            _raise_prediction_domain_error(
                code="unknown_players",
                message="Player snapshots are unavailable for one or more requested players.",
                context={"players": exc.players},
            )
        except NegativeSnapshotAgeError as exc:
            _record_prediction_failure(
                prediction_mode, prediction_start, "negative_snapshot_age"
            )
            _raise_prediction_domain_error(
                code="negative_snapshot_age",
                message="Player snapshot dates must not be later than the match date.",
                context={
                    "match_date": exc.match_date.isoformat(),
                    "affected_players": [
                        {
                            **affected,
                            "as_of_date": affected["as_of_date"].isoformat(),
                        }
                        for affected in exc.affected_players
                    ],
                },
            )
        except StaleSnapshotsError as exc:
            code = "stale_snapshots"
            context = exc.context.model_dump(mode="json")
            _record_prediction_failure(prediction_mode, prediction_start, code)
            get_request_logger().warning(
                "prediction_domain_rejected",
                prediction_mode=prediction_mode,
                sample_data=sample_data,
                code=code,
                context=context,
            )
            _raise_prediction_domain_error(code=code, message=str(exc), context=context)
        except (KeyError, ValueError) as exc:
            _record_prediction_failure(
                prediction_mode, prediction_start, "prediction_failed"
            )
            get_request_logger().warning(
                "prediction_domain_rejected",
                prediction_mode=prediction_mode,
                sample_data=sample_data,
                code="prediction_failed",
                error_type=exc.__class__.__name__,
            )
            _raise_prediction_domain_error(
                code="prediction_failed",
                message="Prediction could not be completed.",
                context={},
            )
        except Exception as exc:
            _record_prediction_failure(
                prediction_mode, prediction_start, exc.__class__.__name__
            )
            get_request_logger().exception(
                "prediction_failed",
                prediction_mode=prediction_mode,
                sample_data=sample_data,
            )
            raise HTTPException(
                status_code=500, detail="Internal prediction engine error."
            ) from exc

    record_prediction_metrics(
        mode=prediction_mode,
        duration_seconds=request_duration_seconds(prediction_start),
    )
    if sample_data:
        record_sample_data_prediction()
    get_request_logger().info(
        "prediction_succeeded",
        prediction_mode=prediction_mode,
        sample_data=sample_data,
        p1_name=p1_name,
        p2_name=p2_name,
        date=date_string,
    )

    predicted_winner = p1_name if predicted_class == 1 else p2_name
    return PredictionResponse(
        predicted_winner=predicted_winner,
        predicted_class=predicted_class,
        p1_win_probability=probability,
        prediction_mode=prediction_mode,
        model=assets.model_identity,
        feature_freshness=feature_freshness,
    )


def create_prediction_router(limiter: Limiter) -> APIRouter:
    """Build prediction routes bound to one application's limiter."""
    router = APIRouter()

    @router.post(
        "/predict/production",
        response_model=PredictionResponse,
        responses=PREDICTION_DOMAIN_RESPONSES,
    )
    @limiter.shared_limit(
        prediction_rate_limit,
        scope="predict",
        key_func=get_prediction_rate_limit_key,
    )
    def predict_match_outcome_production(
        request: Request,
        response: Response,
        prediction_request: PredictionRequest,
        _current_user: CurrentActiveUserDep,
        runtime_assets: RuntimeAssetsDep,
    ) -> PredictionResponse:
        """Run an authenticated production prediction."""
        return _predict_match_outcome(
            request=request,
            runtime_assets=runtime_assets,
            prediction_request=prediction_request,
            prediction_mode="production",
        )

    @router.post(
        "/predict/testing",
        response_model=PredictionResponse,
        responses=PREDICTION_DOMAIN_RESPONSES,
    )
    @limiter.shared_limit(
        prediction_rate_limit,
        scope="predict",
        key_func=get_prediction_rate_limit_key,
    )
    def predict_match_outcome_testing(
        request: Request,
        response: Response,
        prediction_request: PredictionRequest,
        _current_user: CurrentActiveUserDep,
        runtime_assets: RuntimeAssetsDep,
    ) -> PredictionResponse:
        """Run an authenticated non-production sample/testing prediction."""
        if not testing_predictions_enabled():
            raise HTTPException(
                status_code=403,
                detail="Testing predictions are only available in development and test environments.",
            )
        return _predict_match_outcome(
            request=request,
            runtime_assets=runtime_assets,
            prediction_request=prediction_request,
            prediction_mode="testing",
        )

    return router
