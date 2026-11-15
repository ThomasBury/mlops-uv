"""Reusable authentication dependencies for API route signatures."""

from typing import Annotated

from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm

from acebet.app.dependencies.auth import get_current_active_user, get_current_user
from acebet.app.dependencies.data_models import UserInDB

CurrentUserDep = Annotated[UserInDB, Depends(get_current_user)]
CurrentActiveUserDep = Annotated[UserInDB, Depends(get_current_active_user)]
OAuth2FormDep = Annotated[OAuth2PasswordRequestForm, Depends()]
