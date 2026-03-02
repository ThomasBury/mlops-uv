"""Model and data loading helpers for match outcome prediction."""

from pathlib import Path
from typing import Any

import pandas as pd
from joblib import load


def load_data(data_file: str | Path) -> pd.DataFrame:
    """Load match data from a Feather file.

    Args:
        data_file: Path to the Feather dataset.

    Returns:
        pd.DataFrame: Loaded dataset with ``date`` cast to datetime.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If the file cannot be parsed.
    """
    try:
        df = pd.read_feather(data_file)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Data file '{data_file}' not found. Please check the file path."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Error occurred while loading data: {exc}") from exc


def query_data(df: pd.DataFrame, p1_name: str, p2_name: str, date: str) -> pd.DataFrame:
    """Filter rows for a given player pair and match date.

    Args:
        df: Match dataset with ``p1``, ``p2``, and ``date`` columns.
        p1_name: Name of player one.
        p2_name: Name of player two.
        date: Match date in ``YYYY-MM-DD`` format.

    Returns:
        pd.DataFrame: Rows matching the players in either order on the given date.

    Raises:
        KeyError: If required columns are missing from the input frame.
        ValueError: If query execution fails.
    """
    try:
        df["date"] = pd.to_datetime(df["date"])
        match_date = pd.to_datetime(date)
        pair_one = (df["p1"] == p1_name) & (df["p2"] == p2_name)
        pair_two = (df["p1"] == p2_name) & (df["p2"] == p1_name)
        same_date = df["date"] == match_date
        return df[(pair_one | pair_two) & same_date]
    except KeyError as exc:
        raise KeyError(f"Invalid column names in the data: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Error occurred while querying data: {exc}") from exc


def predict(model: Any, df: pd.DataFrame) -> tuple[Any, Any, str]:
    """Run model inference on filtered match rows.

    Args:
        model: Trained estimator exposing ``predict`` and ``predict_proba``.
        df: Filtered dataset containing model features.

    Returns:
        tuple[Any, Any, str]: Predicted probability array, class array, and player name.

    Raises:
        ValueError: If model inference fails.
    """
    predictors = df.columns.drop(
        ["target", "date", "sets_p1", "sets_p2", "b365_p1", "b365_p2", "ps_p1", "ps_p2"]
    )
    X = df[predictors].copy()
    try:
        prob = model.predict_proba(X)[:, 1]
        class_ = model.predict(X)
        return prob, class_, X["p1"].values[0]
    except Exception as exc:
        raise ValueError(f"Error occurred during prediction: {exc}") from exc


def load_model(model_path: str | Path):
    """Load the most recently modified model artifact.

    Args:
        model_path: Directory containing ``model_*.joblib`` files.

    Returns:
        Any: Deserialized trained model.

    Raises:
        ValueError: If no model artifact is found.
    """
    model_files = [file for file in Path(model_path).glob("model_*.joblib")]
    if not model_files:
        raise ValueError(f"No model files found in '{model_path}'.")

    most_recent_model_file = max(model_files, key=lambda file: file.stat().st_mtime)
    return load(most_recent_model_file)


def make_prediction(
    data_file: str | Path, model_path: str | Path, p1_name: str, p2_name: str, date: str
):
    """Load assets and return model predictions for one match specification.

    Args:
        data_file: Path to the feature dataset.
        model_path: Directory containing trained model artifacts.
        p1_name: Name of player one.
        p2_name: Name of player two.
        date: Match date in ``YYYY-MM-DD`` format.

    Returns:
        tuple[Any, Any, str]: Predicted probability array, class array, and player name.

    Raises:
        ValueError: If data loading, query, model loading, or inference fails.
    """
    df = load_data(data_file)
    model = load_model(model_path)
    df_filtered = query_data(df, p1_name, p2_name, date)
    return predict(model, df_filtered)
