"""Build the prepared ATP dataset from tutorial-sized raw CSV inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from acebet.dataprep.dataprep import validate_prepared_data_contract
from acebet.features import add_derived_match_features, elo_win_probability

REQUIRED_RAW_COLUMNS = (
    "date",
    "p1",
    "p2",
    "rank_p1",
    "rank_p2",
    "sets_p1",
    "sets_p2",
    "elo_p1",
    "elo_p2",
)

OPTIONAL_TEXT_DEFAULTS = {
    "location": "Unknown",
    "tournament": "Online Match",
    "series": "ATP250",
    "court": "Outdoor",
    "surface": "Hard",
    "round": "1st Round",
}
OPTIONAL_NUMERIC_DEFAULTS = {
    "atp": 0,
    "best of": 3,
}
OPTIONAL_NULLABLE_TEXT_COLUMNS = ("comment",)
OPTIONAL_NULLABLE_FLOAT_COLUMNS = ("ps_p1", "ps_p2", "b365_p1", "b365_p2")

PREPARED_COLUMN_ORDER = (
    "atp",
    "location",
    "tournament",
    "date",
    "series",
    "court",
    "surface",
    "round",
    "best of",
    "p1",
    "p2",
    "rank_p1",
    "rank_p2",
    "sets_p1",
    "sets_p2",
    "comment",
    "ps_p1",
    "ps_p2",
    "b365_p1",
    "b365_p2",
    "elo_p1",
    "elo_p2",
    "proba_elo",
    "target",
    "year",
    "month",
    "day",
    "rank_diff",
    "best_ranked",
)


def load_raw_matches(input_path: Path) -> pd.DataFrame:
    """Load one raw CSV or a directory of CSV files."""
    if input_path.is_dir():
        csv_paths = sorted(input_path.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(
                f"No CSV files found in raw input directory: {input_path}"
            )
        frames = [pd.read_csv(path) for path in csv_paths]
        return pd.concat(frames, ignore_index=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Raw input not found at {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Raw input must be a CSV file or directory of CSVs: {input_path}"
        )
    return pd.read_csv(input_path)


def _require_raw_columns(df: pd.DataFrame) -> None:
    missing_columns = sorted(set(REQUIRED_RAW_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Raw dataset is missing required columns: {missing_columns}")


def _coerce_numeric(series: pd.Series, column_name: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series)
    except ValueError as exc:
        raise ValueError(
            f"Column {column_name!r} contains non-numeric values."
        ) from exc

    if not np.isfinite(numeric.dropna()).all():
        raise ValueError(f"Column {column_name!r} must contain finite numeric values.")
    return numeric


def normalize_raw_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw CSV rows to the prepared dataset schema."""
    _require_raw_columns(df)
    null_columns = sorted(
        column for column in REQUIRED_RAW_COLUMNS if df[column].isna().any()
    )
    if null_columns:
        raise ValueError(
            f"Raw dataset has null values in required columns: {null_columns}"
        )
    normalized = df.copy()
    normalized["date"] = pd.to_datetime(
        normalized["date"], errors="raise"
    ).dt.normalize()

    for column in ("p1", "p2"):
        normalized[column] = normalized[column].astype(str)

    for column in ("rank_p1", "rank_p2"):
        ranks = _coerce_numeric(normalized[column], column)
        if (ranks <= 0).any():
            raise ValueError("Prepared dataset rank columns must be positive.")
        if (ranks % 1 != 0).any():
            raise ValueError(f"Column {column!r} ranks must be integral.")
        normalized[column] = ranks.astype("int64")
    for column in ("sets_p1", "sets_p2", "elo_p1", "elo_p2"):
        normalized[column] = _coerce_numeric(normalized[column], column).astype(
            "float64"
        )

    for column, default_value in OPTIONAL_NUMERIC_DEFAULTS.items():
        if column in normalized.columns:
            normalized[column] = _coerce_numeric(normalized[column], column).fillna(
                default_value
            )
        else:
            normalized[column] = default_value
        normalized[column] = normalized[column].astype("int64")

    for column, default_value in OPTIONAL_TEXT_DEFAULTS.items():
        if column in normalized.columns:
            normalized[column] = (
                normalized[column]
                .astype("object")
                .where(normalized[column].notna(), default_value)
            )
        else:
            normalized[column] = default_value

    for column in OPTIONAL_NULLABLE_TEXT_COLUMNS:
        if column in normalized.columns:
            normalized[column] = normalized[column].astype("object")
        else:
            normalized[column] = pd.Series([pd.NA] * len(normalized), dtype="object")

    for column in OPTIONAL_NULLABLE_FLOAT_COLUMNS:
        if column in normalized.columns:
            normalized[column] = _coerce_numeric(normalized[column], column).astype(
                "float64"
            )
        else:
            normalized[column] = pd.Series(
                [float("nan")] * len(normalized), dtype="float64"
            )

    normalized["target"] = normalized["sets_p1"] > normalized["sets_p2"]
    normalized["proba_elo"] = normalized.apply(
        lambda row: elo_win_probability(row["elo_p1"], row["elo_p2"]),
        axis=1,
    )
    normalized = add_derived_match_features(normalized)
    normalized = normalized.sort_values("date", kind="stable").reset_index(drop=True)

    missing_prepared_columns = sorted(
        set(PREPARED_COLUMN_ORDER) - set(normalized.columns)
    )
    if missing_prepared_columns:
        raise ValueError(
            f"Prepared dataset is missing columns after normalization: {missing_prepared_columns}"
        )
    return normalized.loc[:, PREPARED_COLUMN_ORDER]


def build_prepared_dataset(
    input_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Load raw CSV data, derive prepared features, and write a feather dataset."""
    prepared = validate_prepared_data_contract(
        normalize_raw_matches(load_raw_matches(input_path))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_feather(output_path)
    return prepared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build the prepared ATP dataset from a raw CSV file or directory of CSVs.",
    )
    parser.add_argument(
        "--input-path", required=True, help="Raw CSV file or directory of CSVs."
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Destination feather path. Required to prevent accidental artifact replacement.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = Path(args.output_path)
    data = build_prepared_dataset(Path(args.input_path), output_path)
    print(f"Built {len(data)} prepared rows at {output_path}")
