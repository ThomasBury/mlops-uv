import argparse
from pathlib import Path

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

    df = pd.read_feather(input_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Prepared dataset is missing columns: {missing_columns}")

    df = add_derived_match_features(df)
    df = df.sort_values("date").reset_index(drop=True)
    df = df.reset_index(drop=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_feather(output_path)

    if player_stats_output_path is not None:
        player_stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        build_latest_player_stats(df).to_feather(player_stats_output_path)

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the prepared ATP dataset.")
    parser.add_argument("--input-path", default=str(DEFAULT_PRODUCTION_DATA_PATH))
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
