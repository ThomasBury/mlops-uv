from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import load
from sklearn.utils.validation import check_is_fitted

from acebet.app.dependencies.data_models import (
    FeatureFreshness,
    ModelIdentity,
    SnapshotPlayer,
    StaleSnapshotsContext,
)
from acebet.features import (
    FeatureContractPipeline,
    PLAYER_STATS_COLUMNS,
    PredictionEstimator,
    align_features_to_model,
    assert_model_artifact_compatible,
    build_match_feature_row,
)
from acebet.model_artifacts import read_validated_model_artifact


@dataclass(frozen=True)
class LoadedModelBundle:
    """Estimator and identity derived from one validated artifact bundle."""

    estimator: PredictionEstimator
    identity: ModelIdentity
    artifact_path: Path
    player_stats_sha256: str | None


@dataclass(frozen=True)
class PlayerSnapshot:
    """Resolved online features for one player."""

    player: str
    rank: float
    elo: float
    as_of_date: date

    def feature_values(self) -> dict[str, float]:
        """Return the snapshot values consumed by feature assembly."""
        return {"rank": self.rank, "elo": self.elo}


class AffectedPlayer(TypedDict):
    """Typed details for one feature snapshot."""

    player: str
    role: Literal["p1", "p2"]
    as_of_date: date
    age_days: int


class UnknownPlayerSnapshotsError(ValueError):
    """Raised when one or more requested player snapshots are unavailable."""

    def __init__(self, players: list[str]) -> None:
        super().__init__("Player snapshots are unavailable.")
        self.players = players


class NegativeSnapshotAgeError(ValueError):
    """Raised when one or more snapshots postdate the requested match."""

    def __init__(
        self, match_date: date, affected_players: list[AffectedPlayer]
    ) -> None:
        super().__init__("Player snapshots must not postdate the match.")
        self.match_date = match_date
        self.affected_players = affected_players


class StaleSnapshotsError(ValueError):
    """Raised when one or more snapshots exceed the configured age limit."""

    def __init__(self, context: StaleSnapshotsContext) -> None:
        super().__init__(
            f"Player snapshots exceed the configured maximum age of {context.max_age_days} days."
        )
        self.context = context


def load_player_stats(
    player_stats_file: str | Path,
    *,
    expected_sha256: str | None = None,
) -> pd.DataFrame:
    """
    Load the feature-store-lite player state table.

    This table contains the latest known Rank and Elo for every player,
    allowing the system to score matches for players without looking up
    historical match rows.

    Parameters
    ----------
    player_stats_file : str or Path
        Path to the latest_player_stats.feather file.
    expected_sha256 : str, optional
        Expected lowercase SHA-256 digest of the player stats Feather file.

    Returns
    -------
    pd.DataFrame
        The player stats table with 'as_of_date' converted to datetime.
    """
    try:
        source = player_stats_file
        if expected_sha256 is not None:
            source = BytesIO(Path(player_stats_file).read_bytes())
            if sha256(source.getbuffer()).hexdigest() != expected_sha256:
                raise ValueError(
                    "Player stats SHA-256 mismatch for the promoted bundle."
                )
        df = pd.read_feather(source)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Player stats file '{player_stats_file}' not found. Please check the file path."
        ) from exc
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error occurred while loading player stats: {e}") from e

    missing_columns = sorted(set(PLAYER_STATS_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Player stats table is missing columns: {missing_columns}")
    if (
        not df["player"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise ValueError("Player identities must be nonblank strings.")
    if df["player"].duplicated().any():
        raise ValueError("Player identities must be unique.")
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce")
    if df["as_of_date"].isna().any():
        raise ValueError("Player snapshot dates must be parseable and non-null.")
    for column in ("rank", "elo"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if df[column].isna().any() or not np.isfinite(df[column]).all():
            raise ValueError(
                f"Player stats {column} must contain finite numeric values."
            )
    if ((df["rank"] <= 0) | (df["rank"] % 1 != 0)).any():
        raise ValueError("Player ranks must be positive integers.")
    if ((df["elo"] <= 0) | (df["elo"] > 5000)).any():
        raise ValueError("Player Elo ratings must be within (0, 5000].")
    return df.set_index("player", drop=False).sort_index()


def predict(
    model: PredictionEstimator, df: pd.DataFrame
) -> tuple[float, Literal[0, 1]]:
    """
    Predict the winning probability and outcome class.

    Parameters
    ----------
    model : Pipeline or BaseEstimator
        The trained model.
    df : pd.DataFrame
        A dataframe containing exactly one match row with all required features.

    Returns
    -------
    tuple[float, int]
        The unmodified class-1/player-1 probability and binary predicted class.
    """
    # Align to the artifact's strict ordered feature contract.
    X = align_features_to_model(model, df)
    raw_probabilities = model.predict_proba(X)
    raw_classes = model.predict(X)
    return normalize_prediction_outputs(raw_probabilities, raw_classes)


def normalize_prediction_outputs(
    raw_probabilities: npt.ArrayLike, raw_classes: npt.ArrayLike
) -> tuple[float, Literal[0, 1]]:
    """Validate exactly one sklearn-shaped binary estimator result."""
    try:
        probabilities = np.asarray(raw_probabilities)
        if probabilities.shape != (1, 2):
            raise ValueError(
                "Estimator predict_proba output must contain one binary row."
            )
        probability_value = probabilities[0, 1]

        classes = np.asarray(raw_classes)
        if classes.shape != (1,):
            raise ValueError("Estimator predict output must contain one class.")
        class_value = classes[0]
        if isinstance(probability_value, (bool, np.bool_)) or isinstance(
            class_value, (bool, np.bool_)
        ):
            raise ValueError("Estimator outputs must be numeric, not boolean.")
        probability = float(probability_value)
        predicted_class_as_float = float(class_value)
    except (IndexError, TypeError, OverflowError) as exc:
        raise ValueError("Estimator returned malformed prediction outputs.") from exc

    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Player-1 win probability must be finite and in [0, 1].")
    if predicted_class_as_float not in (0.0, 1.0):
        raise ValueError("Predicted class must be binary (0 or 1).")
    predicted_class: Literal[0, 1] = 0 if predicted_class_as_float == 0.0 else 1
    return probability, predicted_class


def load_model_bundle(model_path: str | Path) -> LoadedModelBundle:
    """Load an estimator and metadata-derived identity from one validated bundle."""
    artifact = read_validated_model_artifact(model_path)
    metadata = artifact.metadata
    model = load(BytesIO(artifact.model_bytes))
    model_metadata = metadata["model"]
    model_kind = model_metadata["kind"]
    if model_kind == "elo_baseline":
        from acebet.train.estimators import EloBaselineClassifier

        if not isinstance(model, EloBaselineClassifier):
            raise ValueError(
                "Deserialized model type does not match metadata kind 'elo_baseline'."
            )
    elif model_kind == "lightgbm":
        from lightgbm import LGBMClassifier

        if not isinstance(model, FeatureContractPipeline):
            raise ValueError(
                "Deserialized model type does not match metadata kind 'lightgbm'."
            )
        if not isinstance(model.named_steps.get("gbm"), LGBMClassifier):
            raise ValueError("LightGBM artifact is missing its fitted LGBMClassifier.")
    else:
        raise ValueError(f"Unsupported selected model kind {model_kind!r}.")
    assert_model_artifact_compatible(model)
    check_is_fitted(model)
    identity = ModelIdentity(
        kind=model_kind,
        version=Path(model_metadata["filename"]).stem,
        sha256=model_metadata["sha256"],
    )
    return LoadedModelBundle(
        estimator=model,
        identity=identity,
        artifact_path=artifact.path.resolve(),
        player_stats_sha256=metadata.get("player_stats_sha256"),
    )


def resolve_player_snapshots(
    player_stats: pd.DataFrame, p1_name: str, p2_name: str
) -> tuple[PlayerSnapshot, PlayerSnapshot]:
    """Resolve both snapshots exactly once and retain request order on failure."""
    missing_columns = sorted(set(PLAYER_STATS_COLUMNS) - set(player_stats.columns))
    if missing_columns:
        raise ValueError(f"Player stats table is missing columns: {missing_columns}")

    stats_by_player = (
        player_stats
        if player_stats.index.name == "player" and player_stats.index.is_unique
        else player_stats.set_index("player", drop=False, verify_integrity=True)
    )
    missing_players = [
        player for player in (p1_name, p2_name) if player not in stats_by_player.index
    ]
    if missing_players:
        raise UnknownPlayerSnapshotsError(missing_players)

    def snapshot_for(player: str) -> PlayerSnapshot:
        row = stats_by_player.loc[player]
        as_of_timestamp = pd.Timestamp(row["as_of_date"])
        if not isinstance(as_of_timestamp, pd.Timestamp):
            raise ValueError(f"Player snapshot date is missing for {player!r}.")
        as_of_date = as_of_timestamp.date()
        return PlayerSnapshot(
            player=player,
            rank=float(row["rank"]),
            elo=float(row["elo"]),
            as_of_date=as_of_date,
        )

    return snapshot_for(p1_name), snapshot_for(p2_name)


def calculate_feature_freshness(
    match_date: date,
    p1_snapshot: PlayerSnapshot,
    p2_snapshot: PlayerSnapshot,
    *,
    max_age_days: int | None = None,
) -> FeatureFreshness:
    """Calculate calendar-day snapshot ages and reject future or overly stale snapshots."""
    snapshots: tuple[tuple[Literal["p1", "p2"], PlayerSnapshot], ...] = (
        ("p1", p1_snapshot),
        ("p2", p2_snapshot),
    )
    ages = {
        role: (match_date - snapshot.as_of_date).days for role, snapshot in snapshots
    }
    snapshot_details: list[AffectedPlayer] = [
        {
            "player": snapshot.player,
            "role": role,
            "as_of_date": snapshot.as_of_date,
            "age_days": ages[role],
        }
        for role, snapshot in snapshots
    ]
    future_players = [player for player in snapshot_details if player["age_days"] < 0]
    if future_players:
        raise NegativeSnapshotAgeError(match_date, future_players)
    if max_age_days is not None:
        stale_players = [
            SnapshotPlayer(**player)
            for player in snapshot_details
            if player["age_days"] > max_age_days
        ]
        if stale_players:
            raise StaleSnapshotsError(
                StaleSnapshotsContext(
                    match_date=match_date,
                    max_age_days=max_age_days,
                    affected_players=stale_players,
                )
            )
    return FeatureFreshness(
        p1_as_of_date=p1_snapshot.as_of_date,
        p2_as_of_date=p2_snapshot.as_of_date,
        p1_age_days=ages["p1"],
        p2_age_days=ages["p2"],
    )


def make_online_prediction(
    model: PredictionEstimator,
    p1_name: str,
    p2_name: str,
    date: str,
    p1_snapshot: PlayerSnapshot,
    p2_snapshot: PlayerSnapshot,
    series: str,
    court: str,
    surface: str,
    round: str,
    best_of: int,
) -> tuple[float, Literal[0, 1]]:
    """
    Assemble features from player state and match context, then score them.

    This is the "stateless" prediction path. It doesn't need the historical
    match to exist in the database; it builds the feature vector from
    the latest known player stats.

    Parameters
    ----------
    model : object
        Pre-loaded model artifact.
    p1_name : str
        Name of player 1.
    p2_name : str
        Name of player 2.
    date : str
        Match date.

    Notes
    -----
    All five match-context values are required; online scoring has no defaults.
    """
    # Build the feature vector from scratch using current player state
    feature_row = build_match_feature_row(
        p1_name=p1_name,
        p2_name=p2_name,
        date=date,
        p1_snapshot=p1_snapshot.feature_values(),
        p2_snapshot=p2_snapshot.feature_values(),
        series=series,
        court=court,
        surface=surface,
        round=round,
        best_of=best_of,
    )
    return predict(model, feature_row)
