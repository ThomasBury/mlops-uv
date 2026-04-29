from datetime import date
from pydantic import BaseModel


class Token(BaseModel):
    """
    Data model for access tokens.

    Attributes
    ----------
    access_token : str
        The access token.
    token_type : str
        The token type.

    """

    access_token: str
    token_type: str


class TokenData(BaseModel):
    """
    Data model for token data.

    Attributes
    ----------
    username : str or None
        The username associated with the token.

    """

    username: str | None = None


class User(BaseModel):
    """
    Data model for user information.

    Attributes
    ----------
    username : str
        The username of the user.
    email : str or None
        The email address of the user.
    full_name : str or None
        The full name of the user.
    disabled : bool or None
        Whether the user is disabled.

    """

    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    """
    Data model for user information stored in the database.

    Attributes
    ----------
    hashed_password : str
        The hashed password of the user.

    """

    hashed_password: str


class PredictionRequest(BaseModel):
    """
    Data model for prediction requests.

    Parameters
    ----------
    p1_name : str
        The name of player 1.
    p2_name : str
        The name of player 2.
    date : date
        The date of the match.
    testing : bool, optional
        Whether to use the testing asset resolution path.
    atp : int, optional
        ATP points/category for the tournament.
    location : str, optional
        City or country where the match is played.
    tournament : str, optional
        Name of the tournament.
    series : str, optional
        Tournament series (e.g., Grand Slam, ATP500).
    court : str, optional
        Type of court (e.g., Indoor, Outdoor).
    surface : str, optional
        Type of surface (e.g., Hard, Clay, Grass).
    round : str, optional
        The round of the tournament.
    best_of : int, optional
        The number of sets (e.g., 3 or 5).

    Attributes
    ----------
    p1_name : str
        The name of player 1.
    p2_name : str
        The name of player 2.
    date : date
        The date of the match.
    testing : bool
        Whether the testing asset resolution path is used.
    """

    p1_name: str
    p2_name: str
    date: date
    testing: bool = False
    atp: int | None = None
    location: str | None = None
    tournament: str | None = None
    series: str | None = None
    court: str | None = None
    surface: str | None = None
    round: str | None = None
    best_of: int | None = None


class PredictionResponse(BaseModel):
    """
    Data model for prediction responses.

    Attributes
    ----------
    player_name : str or None
        The name of the predicted winning player.
    prob : float or None
        The predicted winning probability.
    class_ : int or None
        The class of the prediction (0 or 1).

    """

    player_name: str | None = None
    prob: float | None = None
    class_: int | None = None
