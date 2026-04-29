import logging
import numpy as np

from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.security import OAuth2PasswordRequestForm
from starlette.types import Message

# Import the prediction logic and data contracts
from acebet.app.dependencies.predict_winner import (
    make_online_prediction,
    load_model,
    load_player_stats,
)
from acebet.app.dependencies.data_models import (
    Token,
    User,
    UserInDB,
    PredictionRequest,
    PredictionResponse,
)
from acebet.app.dependencies.auth import (
    authenticate_user,
    fake_users_db,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_current_active_user,
    get_current_user,
)

from slowapi import Limiter
from slowapi.util import get_remote_address

# Configure standard logging to a file for auditing requests
logging.basicConfig(filename="info.log", level=logging.DEBUG)

# Define paths for asset resolution
PACKAGE_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DATA_DIR = PROJECT_ROOT / "data"
PROJECT_MODEL_DIR = PROJECT_ROOT / "models"
REQUEST_LOG_EXCERPT_BYTES = 4096


def resolve_prediction_assets(testing: bool) -> tuple[Path, Path]:
    """
    Locate the model and player-state files required for online prediction.

    Parameters
    ----------
    testing : bool
        If True, prioritize sample/testing datasets. If False, use production data.

    Returns
    -------
    tuple[Path, Path]
        (player_stats_file_path, model_file_path)

    Raises
    ------
    FileNotFoundError
        If required assets cannot be found in expected locations.
    """
    if testing:
        stats_candidates = [
            PACKAGE_DATA_DIR / "latest_player_stats_sample.feather",
            PACKAGE_DATA_DIR / "latest_player_stats.feather",
            PROJECT_DATA_DIR / "latest_player_stats.feather",
        ]
    else:
        stats_candidates = [
            PROJECT_DATA_DIR / "latest_player_stats.feather",
            PACKAGE_DATA_DIR / "latest_player_stats.feather",
        ]

    stats_file = next((path for path in stats_candidates if path.exists()), None)
    if stats_file is None:
        raise FileNotFoundError(
            f"No player stats table found for testing={testing}. Checked: {stats_candidates}"
        )

    model_candidates = [PROJECT_MODEL_DIR, PACKAGE_DATA_DIR, PROJECT_ROOT]
    model_path = next(
        (path for path in model_candidates if any(path.glob("model_*.joblib"))),
        None,
    )
    if model_path is None:
        raise FileNotFoundError(f"No prediction model found. Checked: {model_candidates}")

    return stats_file, model_path


def load_prediction_assets(
    testing: bool, model_cache: dict[Path, object]
) -> tuple[object, object]:
    """Load and cache the model plus player-state table for one runtime mode."""
    stats_file, model_path = resolve_prediction_assets(testing=testing)
    cache_key = model_path.resolve()
    model = model_cache.get(cache_key)
    if model is None:
        model = load_model(model_path)
        model_cache[cache_key] = model
    player_stats = load_player_stats(stats_file)
    return model, player_stats


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Manager: Executes logic during server startup and shutdown.

    In MLOps, this pattern is used to pre-load large assets (models, reference tables)
    into memory. This avoids expensive disk I/O and deserialization overhead
    on every individual API request.
    """
    model_cache: dict[Path, object] = {}
    try:
        app.state.model, app.state.player_stats = load_prediction_assets(
            testing=False, model_cache=model_cache
        )
        app.state.model_test, app.state.player_stats_test = load_prediction_assets(
            testing=True, model_cache=model_cache
        )
    except Exception:
        logging.exception("CRITICAL: Failed to load assets during startup.")
        raise

    logging.info("ML assets loaded successfully into app.state.")
    try:
        yield
    finally:
        logging.info("Shutting down API server.")


# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["12/minute"])

# Create FastAPI App with Lifespan
app = FastAPI(
    title="AceBet Match Predictor",
    description="MLOps Tutorial API for Tennis Match Winner Prediction",
    lifespan=lifespan,
)

# Attach Limiter to App State
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def log_request_summary(
    method: str, path: str, status_code: int, req_body_excerpt: str | None
) -> None:
    """Log a small request summary without buffering full response bodies."""
    log_line = f"{method} {path} -> {status_code}"
    if req_body_excerpt:
        logging.info("%s | request=%s", log_line, req_body_excerpt)
        return
    logging.info(log_line)


def attach_background_log(
    response: Response, method: str, path: str, status_code: int, req_body_excerpt: str | None
) -> Response:
    """Preserve any existing background task while attaching request logging."""
    tasks = BackgroundTasks()
    if response.background is None:
        response.background = tasks
    elif hasattr(response.background, "tasks"):
        tasks.tasks.extend(response.background.tasks)
    else:
        tasks.tasks.append(response.background)
    tasks.add_task(log_request_summary, method, path, status_code, req_body_excerpt)
    response.background = tasks
    return response


async def set_body(request: Request, body: bytes) -> None:
    """Restore the request stream after middleware reads the body."""

    async def receive() -> Message:
        return {"type": "http.request", "body": body}

    request._receive = receive


@app.middleware("http")
async def user_logging_middleware(request: Request, call_next):
    """
    Middleware to capture a compact request summary for auditing.
    """
    req_body_excerpt: str | None = None
    if request.method in {"POST", "PUT", "PATCH"}:
        req_body = await request.body()
        await set_body(request, req_body)
        if req_body:
            req_body_excerpt = req_body[:REQUEST_LOG_EXCERPT_BYTES].decode(
                errors="replace"
            )
    response = await call_next(request)
    return attach_background_log(
        response=response,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        req_body_excerpt=req_body_excerpt,
    )


@app.get("/")
async def home():
    """Welcome Route."""
    return {"message": "Welcome to the AceBet API!"}


@app.get("/limit/")
@limiter.limit("5/minute")
async def limit(request: Request, user_id: str):
    """Demo route for rate limiting."""
    return {"message": f"API call successful for {user_id}"}


@app.post("/token", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 compatible token login, retrieve an access token for prediction."""
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)):
    """Return the current authenticated user profile."""
    return current_user


@app.get("/users/me/items/", response_model=list[dict])
async def read_own_items(current_user: UserInDB = Depends(get_current_active_user)):
    """Return items owned by the authenticated user."""
    return [{"item_id": "Foo", "owner": current_user.username}]


@app.post("/predict/", response_model=PredictionResponse)
def predict_match_outcome(
    request: Request,
    prediction_request: PredictionRequest,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Authenticated Match Prediction Route.

    NOTE: This is defined as a standard 'def' (synchronous).
    FastAPI will execute this in a separate thread pool. This is a BEST PRACTICE
    for CPU-bound tasks (ML inference) or blocking I/O (Pandas) to avoid
    starving the main Event Loop.

    The route resolves assets from 'app.state' which were pre-loaded during lifespan.
    """
    p1_name = prediction_request.p1_name
    p2_name = prediction_request.p2_name
    testing = prediction_request.testing
    date = prediction_request.date.isoformat()

    if testing:
        model = getattr(request.app.state, "model_test", None)
        player_stats = getattr(request.app.state, "player_stats_test", None)
    else:
        model = getattr(request.app.state, "model", None)
        player_stats = getattr(request.app.state, "player_stats", None)

    if model is None or player_stats is None:
        raise HTTPException(
            status_code=503,
            detail="ML components not initialized. Please check server logs.",
        )

    try:
        match_context = {
            "atp": prediction_request.atp,
            "location": prediction_request.location,
            "tournament": prediction_request.tournament,
            "series": prediction_request.series,
            "court": prediction_request.court,
            "surface": prediction_request.surface,
            "round": prediction_request.round,
            "best_of": prediction_request.best_of,
        }
        prob, class_, player_1 = make_online_prediction(
            model=model,
            player_stats=player_stats,
            p1_name=p1_name,
            p2_name=p2_name,
            date=date,
            match_context=match_context,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logging.error(f"Prediction logic failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal prediction engine error.") from exc

    prob_val = float(np.asarray(prob).ravel()[0])
    class_val = int(np.asarray(class_).ravel()[0])
    prob_percent = round((100 * prob_val), 1)

    return PredictionResponse(player_name=player_1, prob=prob_percent, class_=class_val)
