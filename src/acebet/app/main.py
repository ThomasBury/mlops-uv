"""FastAPI application entrypoint and route handlers."""

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.background import BackgroundTask
from starlette.types import Message

from acebet.app.config import validate_config
from acebet.app.dependencies.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
    fake_users_db,
    get_current_active_user,
    get_current_user,
)
from acebet.app.dependencies.data_models import (
    PredictionRequest,
    PredictionResponse,
    Token,
    User,
    UserInDB,
)
from acebet.app.dependencies.predict_winner import make_prediction

limiter = Limiter(key_func=get_remote_address, default_limits=["12/minute"])
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
logging.basicConfig(filename="info.log", level=logging.DEBUG)


@app.on_event("startup")
def validate_startup_config() -> None:
    """Validate required runtime configuration before serving requests."""
    validate_config()


def log_info(req_body: bytes, res_body: bytes) -> None:
    """Write request and response payloads to the configured logger.

    Args:
        req_body: Raw request body bytes.
        res_body: Raw response body bytes.
    """
    logging.info(req_body)
    logging.info(res_body)


async def set_body(request: Request, body: bytes) -> None:
    """Patch a request so subsequent consumers can read the same body.

    Args:
        request: Incoming request object.
        body: Raw request body bytes previously read from the stream.
    """

    async def receive() -> Message:
        return {"type": "http.request", "body": body}

    request._receive = receive


@app.middleware("http")
async def user_logging_middleware(request: Request, call_next):
    """Capture request/response payloads and log them asynchronously.

    Args:
        request: Incoming request object.
        call_next: FastAPI callback that executes the downstream handler.

    Returns:
        Response: A rebuilt response that preserves the original payload.
    """
    req_body = await request.body()
    await set_body(request, req_body)
    response = await call_next(request)

    res_body = b""
    async for chunk in response.body_iterator:
        res_body += chunk

    task = BackgroundTask(log_info, req_body, res_body)
    return Response(
        content=res_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
        background=task,
    )


@app.get("/")
def home() -> dict[str, str]:
    """Return a health-style welcome payload for the API root."""
    return {"message": "Welcome to the AceBet API!"}


@app.get("/limit/")
@limiter.limit("5/minute")
def limit(request: Request, user_id: str) -> dict[str, str]:
    """Return a response for a rate-limited demonstration endpoint.

    Args:
        request: Incoming request object required by slowapi.
        user_id: Caller-provided identifier used in the response payload.

    Returns:
        dict[str, str]: Confirmation message with the supplied user identifier.
    """
    return {"message": f"API call successful for {user_id}"}


@app.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> dict[str, str]:
    """Authenticate a user and issue a bearer token.

    Args:
        form_data: OAuth2 credential form with username and password.

    Returns:
        dict[str, str]: Access token and token type fields.

    Raises:
        HTTPException: If the submitted credentials are invalid.
    """
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
def read_users_me(
    current_user: UserInDB = Depends(get_current_active_user),
) -> UserInDB:
    """Return the currently authenticated and active user.

    Args:
        current_user: Authenticated user injected by dependency resolution.

    Returns:
        UserInDB: User record for the current access token.
    """
    return current_user


@app.get("/users/me/items/", response_model=list[dict])
def read_own_items(
    current_user: UserInDB = Depends(get_current_active_user),
) -> list[dict[str, str]]:
    """Return a placeholder item list associated with the current user.

    Args:
        current_user: Authenticated user injected by dependency resolution.

    Returns:
        list[dict[str, str]]: Item metadata including owner username.
    """
    return [{"item_id": "Foo", "owner": current_user.username}]


@app.post("/predict/", response_model=PredictionResponse)
def predict_match_outcome(
    request: PredictionRequest, current_user: UserInDB = Depends(get_current_user)
) -> PredictionResponse:
    """Predict match outcome probability for the requested players and date.

    Args:
        request: Input payload containing player names, date, and testing mode.
        current_user: Authenticated user used to protect the endpoint.

    Returns:
        PredictionResponse: Predicted winner, confidence percentage, and class label.
    """
    p1_name = request.p1_name
    p2_name = request.p2_name
    date = request.date
    testing = request.testing

    if testing:
        data_file = (
            Path(__file__).resolve().parents[1] / "data" / "atp_data_sample.feather"
        )
        model_path = Path(__file__).resolve().parents[1] / "data"
    else:
        data_file = (
            Path(__file__).resolve().parents[3] / "data" / "atp_data_production.feather"
        )
        model_path = Path(__file__).resolve().parents[3]

    prob, class_, player_1 = make_prediction(
        data_file=data_file,
        model_path=model_path,
        p1_name=p1_name,
        p2_name=p2_name,
        date=date,
    )
    if isinstance(prob, (list, np.ndarray)):
        prob = np.asarray(prob).ravel()[0]
    prob = float(prob)
    prob = round((100 * prob), 1)

    return PredictionResponse(player_name=player_1, prob=prob, class_=class_)
