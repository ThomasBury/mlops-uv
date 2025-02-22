"""
Authorization and authentication module.
Heavily inspired from https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/
"""

# Let's import all the magical stuff we need!
from datetime import datetime, timedelta  # For handling time
from typing import Annotated, Dict, Union  # For fancy type hints

# Time for FastAPI to shine!
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer  # For authentication
from jose import JWTError, jwt  # For dealing with tokens
from passlib.context import CryptContext  # For password hashing

# Importing the super cool prediction function
from .data_models import TokenData, UserInDB

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Let's have some secret sauce for encryption
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"

# we're using the HS256 algorithm for tokens
ALGORITHM = "HS256"

# Tokens, they typically last only for a short while (in minutes)
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Prepare a database of imaginary users, because we love imagination
fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}

# The secret code to lock and unlock passwords
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# The VIP pass to enter with style
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# The trusty password checker
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
    return pwd_context.verify(plain_password, hashed_password)


# The password hasher
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
    return pwd_context.hash(password)


# The user detective


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


# The secret agent
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


# The token creator
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
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# The security guard checking your identity
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> UserInDB:
    """
    Retrieve the current user based on the provided token.
    This function is responsible for extracting the user's
    information from the JWT token and returning it.
    It is used to authenticate the user and retrieve their user information.
    The token contains a "sub" claim, which typically represents the
    username or user identifier. This function decodes the token
    and extracts this username, then queries the database of users to
    find the corresponding user record.
    This dependency is used in routes that require authentication

    Parameters
    ----------
    token : str
        The JWT token.

    Returns
    -------
    UserInDB
        User information if authenticated, raises an HTTPException if not.
    """
    # Oops, who's that trying to sneak in?
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # The bearer of this token has a story to tell
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    # The guardian checks the database of heroes
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


# The gatekeeper checking your access rights
async def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user),
) -> UserInDB:
    """
    Retrieve the current active user based on the provided user information.
    This function builds on top of get_current_user and adds an additional
    check to see if the user is active or disabled. If the user is marked
    as disabled in the database, this function raises an HTTPException with a
    status code of 400 and a message indicating that the user is inactive.
    This can be used in routes where you want to ensure that only active
    users can access certain resources

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
    """
    Get the user identifier for rate limiting.

    This function returns the username of the current authenticated user,
    which will be used as the identifier for rate limiting.

    Parameters
    ----------
    request : Request
        The FastAPI Request object.
    current_user : UserInDB
        The current authenticated user, obtained from the Depends(get_current_user) dependency.

    Returns
    -------
    str
        The username of the current user as the rate limiting identifier.
    """
    return current_user.username
