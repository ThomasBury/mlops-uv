"""Authentication helpers for the tutorial API."""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Annotated, TypedDict

import bcrypt

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from slowapi.util import get_remote_address

from acebet.settings import get_jwt_signing_key, get_settings

from .data_models import UserInDB

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class UserRecord(TypedDict):
    """Stored fields for one demo user."""

    username: str
    full_name: str
    email: str
    hashed_password: str
    disabled: bool


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain text password against a hashed password.

    Parameters
    ----------
    plain_password : str
        The plain text password to verify.

    hashed_password : str
        The hashed password for comparison.

    Returns
    -------
    bool
        True if the passwords match, False otherwise.
    """
    if len(plain_password.encode("utf-8")) > 72:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except ValueError:
        return False


def get_password_hash(password: str) -> str:
    """
    Generate a hash of a given password.

    Parameters
    ----------
    password : str
        The password to be hashed.

    Returns
    -------
    str
        The hashed password.
    """
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


@lru_cache
def get_demo_users_db() -> dict[str, UserRecord]:
    """Build the demo user store from settings and cache the derived hash."""
    settings = get_settings()
    username = settings.demo_username
    return {
        username: {
            "username": username,
            "full_name": "John Doe",
            "email": f"{username}@example.com",
            "hashed_password": get_password_hash(
                settings.demo_password.get_secret_value()
            ),
            "disabled": False,
        }
    }


def get_user(db: Mapping[str, UserRecord], username: str) -> UserInDB | None:
    """
    Retrieve user information from the database.

    Parameters
    ----------
    db : Dict[str, dict]
        The user database.

    username : str
        The username of the user to retrieve.

    Returns
    -------
    UserInDB
        User information if found, None otherwise.
    """
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(
    fake_db: Mapping[str, UserRecord], username: str, password: str
) -> UserInDB | None:
    """
    Authenticate a user based on provided credentials.

    Parameters
    ----------
    fake_db : Dict[str, dict]
        The fake user database.

    username : str
        The username to authenticate.

    password : str
        The password to authenticate.

    Returns
    -------
    Union[UserInDB, bool]
        User information if authenticated, False otherwise.
    """
    user = get_user(fake_db, username)
    if user is None:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(
    data: Mapping[str, object], expires_delta: timedelta | None = None
) -> str:
    """
    Create an access token.

    Parameters
    ----------
    data : dict
        Data to be encoded into the token.

    expires_delta : timedelta, optional
        Expiry time for the token. Defaults to None.

    Returns
    -------
    str
        The access token.
    """
    to_encode = dict(data)
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    settings = get_settings()
    encoded_jwt = jwt.encode(
        to_encode,
        get_jwt_signing_key(),
        algorithm=settings.jwt_algorithm,
    )
    return encoded_jwt


def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> UserInDB:
    """
    Decode a bearer token and return the matching tutorial user.

    Parameters
    ----------
    token : str
        The JWT token.

    Returns
    -------
    UserInDB
        User information if authenticated, raises an HTTPException if not.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        settings = get_settings()
        payload = jwt.decode(
            token,
            get_jwt_signing_key(),
            algorithms=[settings.jwt_algorithm],
        )
        username = payload.get("sub")
        if not isinstance(username, str) or not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(get_demo_users_db(), username=username)
    if user is None:
        raise credentials_exception
    return user


def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user),
) -> UserInDB:
    """
    Return the authenticated user after checking the disabled flag.

    Parameters
    ----------
    current_user : UserInDB
        User information.

    Returns
    -------
    UserInDB
        Active user information, raises an HTTPException if the user is inactive.
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def get_prediction_rate_limit_key(request: Request) -> str:
    """Return an authenticated-user key for prediction rate limiting."""
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        try:
            settings = get_settings()
            payload = jwt.decode(
                token,
                get_jwt_signing_key(),
                algorithms=[settings.jwt_algorithm],
            )
        except (JWTError, RuntimeError):
            pass
        else:
            username = payload.get("sub")
            if isinstance(username, str) and username:
                return f"user:{username}"
    return f"ip:{get_remote_address(request)}"
