import logging
import numpy as np

from datetime import timedelta
from pathlib import Path
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from fastapi import Depends, FastAPI, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from starlette.background import BackgroundTask
from starlette.types import Message

# Import the prediction function and data models
# from acebet.app.dependencies.logging_user import RouterLoggingMiddleware
from acebet.app.dependencies.predict_winner import make_prediction
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

# from acebet.app.dependencies.logging_user import (
#     log_user_activity,
#     user_activity_middleware,
# )

from slowapi import Limiter
from slowapi.util import get_remote_address

# Configure the rate limiter
# create an instance of the Limiter class and configures it
# to use the get_remote_address function as the key function.
# The key function is responsible for generating a unique identifier
# for rate limiting based on the client's IP address.
limiter = Limiter(key_func=get_remote_address, default_limits=["12/minute"])

# Create an instance of the FastAPI class,
# which serves as the core of your web application.
# This instance is used to define routes, middleware,
# exception handlers, and other configurations for the web service.
app = FastAPI()

# sets the limiter instance you created as a state variable of the FastAPI app.
# This allows you to access the limiter instance from the route functions using app.state.limiter.
app.state.limiter = limiter

# Add an exception handler to the FastAPI app that will catch RateLimitExceeded
# exceptions raised by the slowapi library. The _rate_limit_exceeded_handler function
# handles what should be done when a rate limit is exceeded.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add the user activity middleware to the app
# app.add_middleware(
#     RouterLoggingMiddleware,
#     logger=logging.getLogger(__name__)
# )

# specify the filename of the log file and the logging level.
logging.basicConfig(filename="info.log", level=logging.DEBUG)


# log requests and responses
def log_info(req_body, res_body):
    """
    Logs the request body and the response body.

    Parameters
    ----------
    req_body : bytes
        The request body.
    res_body : bytes
        The response body.
    """
    logging.info(req_body)
    logging.info(res_body)


async def set_body(request: Request, body: bytes):
    """
    Replaces the request's `body` attribute with a function that returns the request body.

    Parameters
    ----------
    request : Request
        The HTTP request object.
    body : bytes
        The request body.
    """

    async def receive() -> Message:
        return {"type": "http.request", "body": body}

    request._receive = receive


@app.middleware("http")
async def user_logging_middleware(request: Request, call_next):
    """
    A middleware function that logs the request body and the response body.

    Parameters
    ----------
    request : Request
        The HTTP request object.
    call_next : Callable
        The next middleware in the chain.

    Returns
    -------
    Response
        The response object.
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


# Home route
@app.get("/")
async def home():
    """
    Welcome Route

    Returns
    -------
    dict
        A welcome message.
    """
    return {"message": "Welcome to the AceBet API!"}


# Rate limiting demonstration route
# Note: the route decorator must be above the limit decorator, not below it
@app.get("/limit/")
@limiter.limit("5/minute")
async def limit(request: Request, user_id: str):
    """
    Rate Limiting Demo Route

    Parameters
    ----------
    user_id : str
        The user identifier.

    Returns
    -------
    dict
        A success message.
    """
    return {"message": f"API call successful for {user_id}"}


# User authentication route
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    User authentication route for generating an access token.

    This function handles the authentication process using the provided
    username and password. If the authentication is successful, an access token
    is generated and returned.

    Parameters
    ----------
    form_data : OAuth2PasswordRequestForm
        The form data containing the username and password.

    Returns
    -------
    dict
        The generated access token along with its type.
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


# User profile route
@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)):
    """
    User Profile Route

    This route returns the profile information of the current active user.

    Parameters
    ----------
    current_user : UserInDB
        The current authenticated user.

    Returns
    -------
    UserInDB
        The user profile information.
    """
    return current_user


@app.get("/users/me/items/", response_model=list[dict])
async def read_own_items(current_user: UserInDB = Depends(get_current_active_user)):
    """
    User's Item Collection Route

    This route returns the item collection of the current active user.

    Parameters
    ----------
    current_user : UserInDB
        The current authenticated user.

    Returns
    -------
    List[dict]
        The list of items owned by the user.
    """
    return [{"item_id": "Foo", "owner": current_user.username}]


# The prediction
# async to facilitate efficient handling of multiple operations
# happening simultaneously, enhancing the responsiveness and performance of the API.
# although the performance is not really important here
# Prediction route with user activity logging
@app.post("/predict/", response_model=PredictionResponse)
# @limiter.limit("60/minute")
async def predict_match_outcome(
    request: PredictionRequest, current_user: UserInDB = Depends(get_current_user)
):
    """
    Prediction Route with Rate Limiting and User Activity Logging

    This route handles match outcome prediction using the provided data.

    Parameters
    ----------
    request : PredictionRequest
        The prediction request data.
    current_user : UserInDB
        The current authenticated user.

    Returns
    -------
    PredictionResponse
        The prediction outcome.
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
    # Ensure prob is a scalar value
    if isinstance(prob, (list, np.ndarray)):
        prob = np.asarray(prob).ravel()[0]
    # Ensure prob is a scalar value
    prob = float(prob)
    prob = round((100 * prob), 1)  # Convert to percentage and round

    return PredictionResponse(player_name=player_1, prob=prob, class_=class_)
