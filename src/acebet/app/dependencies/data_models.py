from pydantic import BaseModel  # For fancy data validation


# Our secret token class to make life easier
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


# Token data for the wise ones
class TokenData(BaseModel):
    """
    Data model for token data.

    Attributes
    ----------
    username : str or None
        The username associated with the token.

    """

    username: str | None = None


# A user model that defines the heroes of our story
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


# The secret agent version of a user
class UserInDB(User):
    """
    Data model for user information stored in the database.

    Attributes
    ----------
    hashed_password : str
        The hashed password of the user.

    """

    hashed_password: str


# The oracle's predictions are in!
class PredictionRequest(BaseModel):
    """
    Data model for prediction requests.

    Parameters
    ----------
    p1_name : str
        The name of player 1.
    p2_name : str
        The name of player 2.
    date : str
        The date of the match in 'YYYY-MM-DD' format.
    testing : bool, optional
        Whether the prediction is for testing purposes, by default False.

    Attributes
    ----------
    p1_name : str
        The name of player 1.
    p2_name : str
        The name of player 2.
    date : str
        The date of the match in 'YYYY-MM-DD' format.
    testing : bool
        Whether the prediction is for testing purposes.

    """

    p1_name: str
    p2_name: str
    date: str
    testing: bool = False


# And the answer is...
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
