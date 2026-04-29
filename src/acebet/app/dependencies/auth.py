"""Authentication helpers for the tutorial API."""

from datetime import datetime, timedelta, timezone
from typing import Annotated, Dict, Union

import bcrypt

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from .data_models import TokenData, UserInDB

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Static JWT configuration used by the tutorial application.
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# In-memory tutorial user store.
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}


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


def get_user(db: Dict[str, dict], username: str) -> Union[UserInDB, None]:
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
    fake_db: Dict[str, dict], username: str, password: str
) -> Union[UserInDB, bool]:
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
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
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
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> UserInDB:
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
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
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


def get_user_identifier(
    request: Request, current_user: UserInDB = Depends(get_current_user)
) -> str:
    """Return the authenticated username for rate-limiting helpers."""
    return current_user.username
