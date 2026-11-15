import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
    validated.loc[:, NUMERIC_CONTRACT_COLUMNS] = numeric_values

    if (numeric_values.loc[:, RANK_COLUMNS] <= 0).any().any():
        raise ValueError("Prepared dataset rank columns must be positive.")

    if (numeric_values.loc[:, RANK_COLUMNS] % 1 != 0).any().any():
        raise ValueError("Prepared dataset rank columns must be integral.")

    if not numeric_values["proba_elo"].between(0, 1).all():
        raise ValueError("Prepared dataset proba_elo must be within [0, 1].")

    target = validated["target"]
    if not pd.api.types.is_bool_dtype(target):
        target_numeric = pd.to_numeric(target, errors="coerce")
        if target_numeric.isna().any() or not target_numeric.isin([0, 1]).all():
            raise ValueError(
                "Prepared dataset target must be boolean or binary 0/1 values."
            )
        validated["target"] = target_numeric.astype("int64")

    return validated


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
            "This tutorial uses data/atp_data_production.feather as its canonical data source."
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
