import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from acebet.features import add_derived_match_features, build_latest_player_stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRODUCTION_DATA_PATH = PROJECT_ROOT / "data" / "atp_data_production.feather"
DEFAULT_PLAYER_STATS_PATH = PROJECT_ROOT / "data" / "latest_player_stats.feather"
REQUIRED_COLUMNS = {
    "date",
    "p1",
    "p2",
    "rank_p1",
    "rank_p2",
    "elo_p1",
    "elo_p2",
    "proba_elo",
    "target",
}
REQUIRED_NON_NULL_COLUMNS = (
    "p1",
    "p2",
    "rank_p1",
    "rank_p2",
    "elo_p1",
    "elo_p2",
    "proba_elo",
    "target",
)
RANK_COLUMNS = ("rank_p1", "rank_p2")
ELO_COLUMNS = ("elo_p1", "elo_p2")
MAX_ELO_RATING = 5000.0
NUMERIC_CONTRACT_COLUMNS = (
    "rank_p1",
    "rank_p2",
    "elo_p1",
    "elo_p2",
    "proba_elo",
)


def validate_prepared_data_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the minimal prepared-data contract before feature derivation."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Prepared dataset is missing columns: {missing_columns}")

    validated = df.copy()
    parsed_dates = pd.to_datetime(validated["date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Prepared dataset contains unparseable date values.")
    validated["date"] = parsed_dates.dt.normalize()

    null_columns = sorted(
        column for column in REQUIRED_NON_NULL_COLUMNS if validated[column].isna().any()
    )
    if null_columns:
        raise ValueError(
            f"Prepared dataset has null values in required columns: {null_columns}"
        )

    numeric_values = validated.loc[:, NUMERIC_CONTRACT_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    non_numeric_columns = sorted(
        column
        for column in NUMERIC_CONTRACT_COLUMNS
        if numeric_values[column].isna().any()
    )
    if non_numeric_columns:
        raise ValueError(
            "Prepared dataset has non-numeric values in required numeric "
            f"columns: {non_numeric_columns}"
        )
    if not np.isfinite(numeric_values).all().all():
        raise ValueError("Prepared dataset numeric columns must contain finite values.")
    validated[numeric_values.columns] = numeric_values

    if (numeric_values.loc[:, RANK_COLUMNS] <= 0).any().any():
        raise ValueError("Prepared dataset rank columns must be positive.")

    if (numeric_values.loc[:, RANK_COLUMNS] % 1 != 0).any().any():
        raise ValueError("Prepared dataset rank columns must be integral.")

    if (
        (
            (numeric_values.loc[:, ELO_COLUMNS] <= 0)
            | (numeric_values.loc[:, ELO_COLUMNS] > MAX_ELO_RATING)
        )
        .any()
        .any()
    ):
        raise ValueError("Prepared dataset Elo columns must be within (0, 5000].")

    if not numeric_values["proba_elo"].between(0, 1).all():
        raise ValueError("Prepared dataset proba_elo must be within [0, 1].")

    elo = numeric_values.loc[:, ELO_COLUMNS].astype(float)
    expected_proba = 1.0 / (1.0 + 10.0 ** ((elo["elo_p2"] - elo["elo_p1"]) / 400.0))
    if not np.allclose(
        numeric_values["proba_elo"], expected_proba, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            "Prepared dataset proba_elo must match the Elo win probability formula."
        )

    target = validated["target"]
    if not pd.api.types.is_bool_dtype(target):
        target_numeric = pd.to_numeric(target, errors="coerce")
        if target_numeric.isna().any() or not target_numeric.isin([0, 1]).all():
            raise ValueError(
                "Prepared dataset target must be boolean or binary 0/1 values."
            )
        validated["target"] = target_numeric.astype("int64")

    return validated


def summarize_elo_causal_evidence(df: pd.DataFrame) -> dict[str, float | int]:
    """Summarize sequential Elo transition ordering and pre-match odds alignment."""
    validated = (
        validate_prepared_data_contract(df)
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    validated["row_id"] = np.arange(len(validated))
    target_bool = validated["target"].astype(bool)

    p1 = validated[["row_id", "date", "p1", "elo_p1"]].rename(
        columns={"p1": "player", "elo_p1": "elo"}
    )
    p1["won"] = target_bool
    p2 = validated[["row_id", "date", "p2", "elo_p2"]].rename(
        columns={"p2": "player", "elo_p2": "elo"}
    )
    p2["won"] = ~target_bool

    appearances = pd.concat([p1, p2], ignore_index=True).sort_values(
        "row_id", kind="stable"
    )
    appearances["ambiguous_date"] = appearances.duplicated(
        ["player", "date"], keep=False
    )
    next_appearance = appearances.groupby("player")[
        ["date", "elo", "won", "ambiguous_date"]
    ].shift(-1)
    appearances["delta_to_next"] = next_appearance["elo"] - appearances["elo"]
    appearances["next_won"] = next_appearance["won"]
    transitions = appearances.loc[
        (next_appearance["date"] > appearances["date"])
        & ~appearances["ambiguous_date"]
        & next_appearance["ambiguous_date"].eq(False)
    ]
    wins = transitions.loc[transitions["won"], "delta_to_next"]
    losses = transitions.loc[~transitions["won"], "delta_to_next"]
    next_wins = transitions.loc[transitions["next_won"].eq(True), "delta_to_next"]
    next_losses = transitions.loc[transitions["next_won"].eq(False), "delta_to_next"]

    summary: dict[str, float | int] = {
        "player_transitions": int(len(transitions)),
        "post_win_elo_increase_rate": float((wins > 0).mean()) if len(wins) else 0.0,
        "post_loss_elo_decrease_rate": (
            float((losses < 0).mean()) if len(losses) else 0.0
        ),
        "elo_increase_by_next_win_rate": float((next_wins > 0).mean())
        if len(next_wins)
        else 0.0,
        "elo_decrease_by_next_loss_rate": float((next_losses < 0).mean())
        if len(next_losses)
        else 0.0,
        "pinnacle_odds_rows": 0,
    }

    if {"ps_p1", "ps_p2"}.issubset(validated.columns):
        odds = validated.copy()
        odds["ps_p1"] = pd.to_numeric(odds["ps_p1"], errors="coerce")
        odds["ps_p2"] = pd.to_numeric(odds["ps_p2"], errors="coerce")
        odds = odds.loc[
            odds["ps_p1"].gt(1.0)
            & odds["ps_p2"].gt(1.0)
            & np.isfinite(odds["ps_p1"])
            & np.isfinite(odds["ps_p2"])
        ]
        if len(odds) >= 2 and odds["target"].nunique() == 2:
            inv_p1 = 1.0 / odds["ps_p1"].to_numpy(dtype=float)
            inv_p2 = 1.0 / odds["ps_p2"].to_numpy(dtype=float)
            pinnacle_proba = inv_p1 / (inv_p1 + inv_p2)
            elo_proba = odds["proba_elo"].to_numpy(dtype=float)
            y_true = odds["target"].to_numpy(dtype=int)
            summary["pinnacle_odds_rows"] = int(len(odds))
            summary["elo_log_loss_on_odds_subset"] = float(
                log_loss(y_true, elo_proba, labels=[0, 1])
            )
            summary["pinnacle_log_loss_on_odds_subset"] = float(
                log_loss(y_true, pinnacle_proba, labels=[0, 1])
            )
            if np.ptp(elo_proba) and np.ptp(pinnacle_proba):
                summary["elo_pinnacle_probability_correlation"] = float(
                    np.corrcoef(elo_proba, pinnacle_proba)[0, 1]
                )
    return summary


def prepare_data(
    input_path: Path = DEFAULT_PRODUCTION_DATA_PATH,
    output_path: Path | None = None,
    player_stats_output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load and validate the prepared ATP dataset used by the tutorial.

    Returns
    -------
    df : pandas.DataFrame
        The prepared data.

    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found at {input_path}. "
            "This tutorial uses data/atp_data_production.feather as its canonical data source. "
            "Run from a repository checkout or supply an explicit --input-path / --data-path."
        )

    df = validate_prepared_data_contract(pd.read_feather(input_path))

    df = add_derived_match_features(df)
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_feather(output_path)

    if player_stats_output_path is not None:
        player_stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        build_latest_player_stats(df).to_feather(player_stats_output_path)

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the prepared ATP dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-path",
        default=str(DEFAULT_PRODUCTION_DATA_PATH),
        help="Prepared Feather dataset to validate.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional destination for a normalized feather copy.",
    )
    parser.add_argument(
        "--player-stats-output-path",
        default=None,
        help=(
            "Optional destination for a feature-store-lite player stats table. "
            f"Example: {DEFAULT_PLAYER_STATS_PATH}"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = Path(args.output_path) if args.output_path else None
    player_stats_output_path = (
        Path(args.player_stats_output_path) if args.player_stats_output_path else None
    )
    data = prepare_data(Path(args.input_path), output_path, player_stats_output_path)
    print(f"Validated {len(data)} rows from {Path(args.input_path)}")
    if player_stats_output_path is not None:
        print(f"Wrote player stats to {player_stats_output_path}")
