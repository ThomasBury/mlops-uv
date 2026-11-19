"""Pydantic contracts exposed by the AceBet API."""

from __future__ import annotations

from datetime import date
from enum import Enum, IntEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError


class Token(BaseModel):
    """OAuth2 access token."""

    access_token: str
    token_type: str


class User(BaseModel):
    """Public user information."""

    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    """User information stored by the demo authentication backend."""

    hashed_password: str


class TournamentSeries(str, Enum):
    """Tournament series represented in the fitted feature contract."""

    ATP250 = "ATP250"
    ATP500 = "ATP500"
    GRAND_SLAM = "Grand Slam"
    INTERNATIONAL = "International"
    INTERNATIONAL_GOLD = "International Gold"
    MASTERS = "Masters"
    MASTERS_1000 = "Masters 1000"
    MASTERS_CUP = "Masters Cup"


class Court(str, Enum):
    """Match court setting."""

    INDOOR = "Indoor"
    OUTDOOR = "Outdoor"


class Surface(str, Enum):
    """Match playing surface."""

    CARPET = "Carpet"
    CLAY = "Clay"
    GRASS = "Grass"
    HARD = "Hard"


class MatchRound(str, Enum):
    """Tournament round represented in the fitted feature contract."""

    FIRST = "1st Round"
    SECOND = "2nd Round"
    THIRD = "3rd Round"
    FOURTH = "4th Round"
    QUARTERFINALS = "Quarterfinals"
    ROUND_ROBIN = "Round Robin"
    SEMIFINALS = "Semifinals"
    FINAL = "The Final"


class BestOf(IntEnum):
    """Supported maximum set counts."""

    THREE = 3
    FIVE = 5


PREDICTION_CONTEXT_FIELDS = ("series", "court", "surface", "round", "best_of")
PREDICTION_CONTEXT_ALLOWED_VALUES: dict[str, tuple[str | int, ...]] = {
    "series": tuple(member.value for member in TournamentSeries),
    "court": tuple(member.value for member in Court),
    "surface": tuple(member.value for member in Surface),
    "round": tuple(member.value for member in MatchRound),
    "best_of": tuple(member.value for member in BestOf),
}

PlayerName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PredictionRequest(BaseModel):
    """Complete, typed match context required for synchronous inference."""

    model_config = ConfigDict(extra="forbid")

    p1_name: PlayerName
    p2_name: PlayerName
    date: date
    series: TournamentSeries
    court: Court
    surface: Surface
    round: MatchRound
    best_of: BestOf

    @field_validator(*PREDICTION_CONTEXT_FIELDS, mode="before")
    @classmethod
    def reject_unsupported_context(cls, value: object, info: ValidationInfo) -> object:
        """Tag well-formed unsupported values for route-specific error mapping."""
        field = info.field_name
        if field is None:
            return value
        allowed_values = PREDICTION_CONTEXT_ALLOWED_VALUES[field]
        expected_type = int if field == "best_of" else str
        is_expected_type = isinstance(value, expected_type)
        if field == "best_of" and isinstance(value, bool):
            is_expected_type = False
        if field == "best_of" and not is_expected_type:
            raise ValueError("best_of must be an integer")
        if is_expected_type and value not in allowed_values:
            raise PydanticCustomError(
                "unsupported_context_value",
                "Unsupported match context value",
                {
                    "field": field,
                    "submitted_value": value,
                    "allowed_values": list(allowed_values),
                },
            )
        return value

    @model_validator(mode="after")
    def reject_equal_players(self) -> PredictionRequest:
        """Require distinct case-sensitive player identities after trimming."""
        if self.p1_name == self.p2_name:
            raise PydanticCustomError(
                "equal_players",
                "Player names must be distinct",
                {"player": self.p1_name},
            )
        return self


class ModelIdentity(BaseModel):
    """Identity read from a validated schema-v2 model bundle."""

    kind: Literal["elo_baseline", "lightgbm"]
    version: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FeatureFreshness(BaseModel):
    """Calendar-day age of each player's online feature snapshot."""

    p1_as_of_date: date
    p2_as_of_date: date
    p1_age_days: int = Field(ge=0)
    p2_age_days: int = Field(ge=0)


class PredictionResponse(BaseModel):
    """Final prediction response contract."""

    predicted_winner: str
    predicted_class: Literal[0, 1]
    p1_win_probability: float = Field(ge=0, le=1)
    prediction_mode: Literal["production", "testing"]
    model: ModelIdentity
    feature_freshness: FeatureFreshness


class EqualPlayersContext(BaseModel):
    """Context for an equal-player validation failure."""

    player: str


class UnknownPlayersContext(BaseModel):
    """Context for missing player snapshots."""

    players: list[str]


class UnsupportedContextValue(BaseModel):
    """One rejected match-context value."""

    field: Literal["series", "court", "surface", "round", "best_of"]
    submitted_value: str | int
    allowed_values: list[str | int]


class UnsupportedContextValuesContext(BaseModel):
    """Context for all unsupported values in request-field order."""

    invalid_fields: list[UnsupportedContextValue]


class SnapshotPlayer(BaseModel):
    """Identity and age of a player snapshot."""

    player: str
    role: Literal["p1", "p2"]
    as_of_date: date
    age_days: int


class NegativeSnapshotPlayer(SnapshotPlayer):
    """One player snapshot newer than the requested match date."""

    age_days: int = Field(lt=0)


class NegativeSnapshotAgeContext(BaseModel):
    """Context for every future-dated player snapshot."""

    match_date: date
    affected_players: list[NegativeSnapshotPlayer]


class StaleSnapshotsContext(BaseModel):
    """Context for snapshots older than the configured limit."""

    match_date: date
    max_age_days: int = Field(gt=0)
    affected_players: list[SnapshotPlayer]


class PredictionFailedContext(BaseModel):
    """Sanitized context for expected scoring failures."""

    model_config = ConfigDict(extra="forbid")


class EqualPlayersError(BaseModel):
    """Equal-player prediction-domain error."""

    code: Literal["equal_players"]
    message: str
    context: EqualPlayersContext


class UnknownPlayersError(BaseModel):
    """Unknown-player prediction-domain error."""

    code: Literal["unknown_players"]
    message: str
    context: UnknownPlayersContext


class UnsupportedContextValuesError(BaseModel):
    """Unsupported-context prediction-domain error."""

    code: Literal["unsupported_context_values"]
    message: str
    context: UnsupportedContextValuesContext


class NegativeSnapshotAgeError(BaseModel):
    """Future-snapshot prediction-domain error."""

    code: Literal["negative_snapshot_age"]
    message: str
    context: NegativeSnapshotAgeContext


class StaleSnapshotsError(BaseModel):
    """Stale-snapshot prediction-domain error."""

    code: Literal["stale_snapshots"]
    message: str
    context: StaleSnapshotsContext


class PredictionFailedError(BaseModel):
    """Expected feature/scoring prediction-domain error."""

    code: Literal["prediction_failed"]
    message: str
    context: PredictionFailedContext


PredictionDomainError = (
    EqualPlayersError
    | UnknownPlayersError
    | UnsupportedContextValuesError
    | NegativeSnapshotAgeError
    | StaleSnapshotsError
    | PredictionFailedError
)


class PredictionDomainErrorResponse(BaseModel):
    """HTTP wrapper for structured prediction-domain failures."""

    detail: PredictionDomainError
