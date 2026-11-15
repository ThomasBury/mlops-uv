"""Authentication and current-user routes."""

from datetime import timedelta

from fastapi import APIRouter, HTTPException, Request, Response, status
from slowapi import Limiter

from acebet.app.dependencies.api import CurrentActiveUserDep, OAuth2FormDep
from acebet.app.dependencies.auth import (
    authenticate_user,
    create_access_token,
    get_demo_users_db,
)
from acebet.app.dependencies.data_models import Token, User, UserInDB
from acebet.settings import get_settings


def token_rate_limit() -> str:
    """Return the configured token route rate limit."""
    return get_settings().token_rate_limit


def create_authentication_router(limiter: Limiter) -> APIRouter:
    """Build authentication routes bound to one application's limiter."""
    router = APIRouter()

    @router.post("/token", response_model=Token)
    @limiter.limit(token_rate_limit)
    def login_for_access_token(
        request: Request,
        response: Response,
        form_data: OAuth2FormDep,
    ) -> Token:
        """Authenticate demo credentials and return an OAuth2 bearer token."""
        user = authenticate_user(
            get_demo_users_db(), form_data.username, form_data.password
        )
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        settings = get_settings()
        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        return Token(access_token=access_token, token_type="bearer")

    @router.get("/users/me", response_model=User)
    def read_users_me(current_user: CurrentActiveUserDep) -> UserInDB:
        """Return the current authenticated user profile."""
        return current_user

    return router
