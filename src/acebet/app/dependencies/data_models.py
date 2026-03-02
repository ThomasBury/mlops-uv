"""Pydantic models used by authentication and prediction endpoints."""

from pydantic import BaseModel


class Token(BaseModel):
    """Response model for bearer token issuance.

    Attributes:
        access_token: JWT token string.
        token_type: OAuth token type (for example, ``"bearer"``).
    """

    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Claims extracted from a decoded token payload.

    Attributes:
        username: Username claim when present.
    """

    username: str | None = None


class User(BaseModel):
    """Public representation of a user account.

    Attributes:
        username: Unique account username.
        email: Optional account email address.
        full_name: Optional full display name.
        disabled: Flag indicating whether access is disabled.
    """

    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    """Internal user representation including authentication metadata.

    Attributes:
        hashed_password: Password hash stored for verification.
    """

    hashed_password: str


class PredictionRequest(BaseModel):
    """Request payload for match outcome prediction.

    Attributes:
        p1_name: Player name in first position.
        p2_name: Player name in second position.
        date: Match date in ``YYYY-MM-DD`` format.
        testing: Whether to use test assets instead of production assets.
    """

    p1_name: str
    p2_name: str
    date: str
    testing: bool = False


class PredictionResponse(BaseModel):
    """Response payload returned by the prediction endpoint.

    Attributes:
        player_name: Predicted winner name.
        prob: Predicted win probability as a percentage.
        class_: Predicted class label.
    """

    player_name: str | None = None
    prob: float | None = None
    class_: int | None = None
