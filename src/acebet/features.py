"""Shared feature contract for training and online prediction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

DATE_COLUMN = "date"

NUMERIC_FEATURE_COLUMNS = (
    "best of",
    "rank_p1",
    "rank_p2",
    "elo_p1",
    "elo_p2",
    "proba_elo",
    "year",
    "month",
    "day",
    "rank_diff",
)
CATEGORICAL_FEATURE_COLUMNS = (
    "p1",
    "p2",
    "series",
    "court",
    "surface",
    "round",
    "best_ranked",
)
MODEL_FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS


class PredictionEstimator(Protocol):
    """Structural contract required by AceBet prediction paths."""

    def predict(self, X: pd.DataFrame) -> npt.ArrayLike:
        """Return one class per input row."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> npt.ArrayLike:
        """Return binary class probabilities."""
        ...


def ordinal_codes_as_categories(encoded: pd.DataFrame) -> pd.DataFrame:
    """Mark ordinal codes as unordered categoricals for native LightGBM splits."""
    return encoded.astype("category")


PLAYER_STATS_COLUMNS = ("player", "rank", "elo", "as_of_date")


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the exact ordered model allowlist and reject missing features."""
    missing_columns = [
        column for column in MODEL_FEATURE_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Model feature contract is missing required columns: "
            f"{missing_columns}. Required ordered columns: "
            f"{list(MODEL_FEATURE_COLUMNS)}."
        )
    return df.loc[:, MODEL_FEATURE_COLUMNS].copy()


class FeatureContractPipeline(Pipeline):
    """Pipeline that enforces the shared contract at every model entrypoint."""

    def fit(
        self, X: pd.DataFrame, y: npt.ArrayLike | None = None, **params: Any
    ) -> FeatureContractPipeline:
        """Select and validate contract columns before fitting."""
        return super().fit(select_model_features(X), y, **params)

    def predict(self, X: pd.DataFrame, **params: Any) -> np.ndarray:
        """Select and validate contract columns before class prediction."""
        return super().predict(select_model_features(X), **params)

    def predict_proba(self, X: pd.DataFrame, **params: Any) -> np.ndarray:
        """Select and validate contract columns before probability prediction."""
        return super().predict_proba(select_model_features(X), **params)


def _model_feature_columns(model: PredictionEstimator) -> list[str] | None:
    """Return fitted model feature names when the artifact exposes them."""
    expected_columns = getattr(model, "feature_names_in_", None)
    if expected_columns is None and isinstance(model, Pipeline):
        preprocessor = model.named_steps.get("preprocessor")
        expected_columns = getattr(preprocessor, "feature_names_in_", None)

    if expected_columns is None:
        return None
    return list(expected_columns)


def assert_model_artifact_compatible(model: PredictionEstimator) -> None:
    """Reject artifacts that do not implement the current feature contract."""
    expected_columns = _model_feature_columns(model)
    contract_columns = list(MODEL_FEATURE_COLUMNS)
    if expected_columns is None:
        raise ValueError(
            "Model artifact is incompatible: it does not expose fitted feature "
            f"names for the required contract {contract_columns}."
        )
    if expected_columns != contract_columns:
        raise ValueError(
            "Model artifact feature contract mismatch: expected ordered columns "
            f"{contract_columns}, found {expected_columns}. Retrain the artifact "
            "with the current preprocessing pipeline."
        )

    if not isinstance(model, Pipeline):
        if getattr(model, "n_features_in_", None) != len(MODEL_FEATURE_COLUMNS):
            raise ValueError(
                "Model artifact feature count does not match the current contract."
            )
        return

    preprocessor = model.named_steps.get("preprocessor")
    if not isinstance(preprocessor, ColumnTransformer):
        raise ValueError(
            "Model artifact preprocessing mismatch: expected a fitted "
            "ColumnTransformer with numeric passthrough and categorical-only "
            "OrdinalEncoder. Legacy whole-frame OrdinalEncoder artifacts must "
            "be retrained."
        )

    transformed_columns = list(preprocessor.get_feature_names_out())
    if transformed_columns != contract_columns:
        raise ValueError(
            "Model artifact transformed feature mismatch: expected ordered "
            f"columns {contract_columns}, found {transformed_columns}."
        )

    output_indices = getattr(preprocessor, "output_indices_", {})
    expected_numeric_slice = slice(0, len(NUMERIC_FEATURE_COLUMNS))
    expected_categorical_slice = slice(
        len(NUMERIC_FEATURE_COLUMNS), len(MODEL_FEATURE_COLUMNS)
    )
    if (
        output_indices.get("numeric") != expected_numeric_slice
        or output_indices.get("categorical") != expected_categorical_slice
    ):
        raise ValueError(
            "Model artifact preprocessing order mismatch: numeric features must "
            "precede categorical features."
        )

    categorical_transformer = preprocessor.named_transformers_.get("categorical")
    categorical_steps = getattr(categorical_transformer, "named_steps", {})
    converter = categorical_steps.get("as_category")
    if (
        not isinstance(categorical_steps.get("ordinal"), OrdinalEncoder)
        or not isinstance(converter, FunctionTransformer)
        or converter.func is not ordinal_codes_as_categories
    ):
        raise ValueError(
            "Model artifact preprocessing mismatch: categorical features must "
            "use OrdinalEncoder followed by the as_category conversion "
            "before LightGBM native categorical handling."
        )


def align_features_to_model(
    model: PredictionEstimator, features: pd.DataFrame
) -> pd.DataFrame:
    """Validate the fitted artifact and return strict ordered input features."""
    assert_model_artifact_compatible(model)
    return select_model_features(features)


def add_derived_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic features shared by training data prep and online scoring."""
    features = df.copy()
    features[DATE_COLUMN] = pd.to_datetime(features[DATE_COLUMN])
    features["year"] = features[DATE_COLUMN].dt.year
    features["month"] = features[DATE_COLUMN].dt.month
    features["day"] = features[DATE_COLUMN].dt.day
    features["rank_diff"] = features["rank_p1"] - features["rank_p2"]
    features["best_ranked"] = "p1"
    features.loc[features["rank_diff"] > 0, "best_ranked"] = "p2"
    return features


def elo_win_probability(elo_p1: float, elo_p2: float) -> float:
    """Return the Elo probability of player 1 beating player 2."""
    exponent = (elo_p2 - elo_p1) / 400
    if exponent > 300:
        return 0.0
    return float(1 / (1 + 10**exponent))


def build_latest_player_stats(matches: pd.DataFrame) -> pd.DataFrame:
    """Build a compact player-state table from the latest row for each player."""
    required_columns = {
        DATE_COLUMN,
        "p1",
        "p2",
        "rank_p1",
        "rank_p2",
        "elo_p1",
        "elo_p2",
    }
    missing_columns = sorted(required_columns - set(matches.columns))
    if missing_columns:
        raise ValueError(
            f"Cannot build player stats, missing columns: {missing_columns}"
        )

    prepared = matches.copy()
    prepared[DATE_COLUMN] = pd.to_datetime(prepared[DATE_COLUMN])
    p1_stats = prepared[[DATE_COLUMN, "p1", "rank_p1", "elo_p1"]].rename(
        columns={
            DATE_COLUMN: "as_of_date",
            "p1": "player",
            "rank_p1": "rank",
            "elo_p1": "elo",
        }
    )
    p2_stats = prepared[[DATE_COLUMN, "p2", "rank_p2", "elo_p2"]].rename(
        columns={
            DATE_COLUMN: "as_of_date",
            "p2": "player",
            "rank_p2": "rank",
            "elo_p2": "elo",
        }
    )
    stats = pd.concat([p1_stats, p2_stats], ignore_index=True)
    stats = stats.dropna(subset=["player", "rank", "elo"])
    stats = stats.sort_values(["player", "as_of_date"]).groupby("player").tail(1)
    stats = stats.sort_values("player").reset_index(drop=True)
    return stats.loc[:, PLAYER_STATS_COLUMNS]


def build_match_feature_row(
    p1_name: str,
    p2_name: str,
    date: str,
    p1_snapshot: Mapping[str, float],
    p2_snapshot: Mapping[str, float],
    series: str,
    court: str,
    surface: str,
    round: str,
    best_of: int,
) -> pd.DataFrame:
    """Assemble one online feature row from resolved snapshots and context."""
    row = {
        "series": series,
        "court": court,
        "surface": surface,
        "round": round,
        "best of": best_of,
        DATE_COLUMN: date,
        "p1": p1_name,
        "p2": p2_name,
        "rank_p1": p1_snapshot["rank"],
        "rank_p2": p2_snapshot["rank"],
        "elo_p1": p1_snapshot["elo"],
        "elo_p2": p2_snapshot["elo"],
    }
    row["proba_elo"] = elo_win_probability(row["elo_p1"], row["elo_p2"])
    derived = add_derived_match_features(pd.DataFrame([row]))
    return select_model_features(derived)
