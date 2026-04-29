"""Shared feature contract for training and online prediction."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

TARGET_COLUMN = "target"
DATE_COLUMN = "date"

# Columns that are unavailable before a match finishes or are operational labels.
NON_FEATURE_COLUMNS = (
    TARGET_COLUMN,
    DATE_COLUMN,
    "sets_p1",
    "sets_p2",
    "b365_p1",
    "b365_p2",
    "ps_p1",
    "ps_p2",
    "comment",
)

PLAYER_STATS_COLUMNS = ("player", "rank", "elo", "as_of_date")

DEFAULT_MATCH_CONTEXT = {
    "atp": 0,
    "location": "Unknown",
    "tournament": "Online Match",
    "series": "ATP250",
    "court": "Outdoor",
    "surface": "Hard",
    "round": "1st Round",
    "best of": 3,
}


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only columns allowed at prediction time."""
    drop_columns: Iterable[str] = [
        column for column in NON_FEATURE_COLUMNS if column in df.columns
    ]
    return df.drop(columns=drop_columns).copy()


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
    return float(1 / (1 + 10 ** ((elo_p2 - elo_p1) / 400)))


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
        raise ValueError(f"Cannot build player stats, missing columns: {missing_columns}")

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
    stats_context: pd.DataFrame,
    match_context: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Assemble a single online prediction row from player state and match context."""
    missing_columns = sorted(set(PLAYER_STATS_COLUMNS) - set(stats_context.columns))
    if missing_columns:
        raise ValueError(f"Player stats table is missing columns: {missing_columns}")

    if stats_context.index.name == "player":
        stats_by_player = stats_context
    else:
        stats_by_player = stats_context.set_index("player", drop=False)
    missing_players = [
        player for player in (p1_name, p2_name) if player not in stats_by_player.index
    ]
    if missing_players:
        raise ValueError(f"Player stats not found for: {missing_players}")

    p1_stats = stats_by_player.loc[p1_name]
    p2_stats = stats_by_player.loc[p2_name]
    context = DEFAULT_MATCH_CONTEXT.copy()
    if match_context:
        context.update(
            {
                ("best of" if key == "best_of" else key): value
                for key, value in match_context.items()
                if value is not None
            }
        )

    row = {
        **context,
        DATE_COLUMN: date,
        "p1": p1_name,
        "p2": p2_name,
        "rank_p1": p1_stats["rank"],
        "rank_p2": p2_stats["rank"],
        "elo_p1": p1_stats["elo"],
        "elo_p2": p2_stats["elo"],
    }
    row["proba_elo"] = elo_win_probability(row["elo_p1"], row["elo_p2"])
    return add_derived_match_features(pd.DataFrame([row]))
