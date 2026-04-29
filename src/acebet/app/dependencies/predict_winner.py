import pandas as pd
from pathlib import Path
from joblib import load

from acebet.features import build_match_feature_row, select_model_features


def load_data(data_file: str | Path) -> pd.DataFrame:
    """
    Load the historical match data from a feather file.

    Parameters
    ----------
    data_file : str or Path
        The path to the historical data file.

    Returns
    -------
    pd.DataFrame
        The loaded and date-parsed data.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    ValueError
        If an error occurs during data loading.
    """
    try:
        # Read data from a feather file and convert the 'date' column to datetime format.
        df = pd.read_feather(data_file)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except FileNotFoundError:
        # Raise an error if the specified data file is not found.
        raise FileNotFoundError(
            f"Data file '{data_file}' not found. Please check the file path."
        )
    except Exception as e:
        # Raise an error for any other loading-related exceptions.
        raise ValueError(f"Error occurred while loading data: {e}")


def load_player_stats(player_stats_file: str | Path) -> pd.DataFrame:
    """
    Load the feature-store-lite player state table.

    This table contains the latest known Rank and Elo for every player,
    allowing the system to score matches for players without looking up
    historical match rows.

    Parameters
    ----------
    player_stats_file : str or Path
        Path to the latest_player_stats.feather file.

    Returns
    -------
    pd.DataFrame
        The player stats table with 'as_of_date' converted to datetime.
    """
    try:
        df = pd.read_feather(player_stats_file)
        df["as_of_date"] = pd.to_datetime(df["as_of_date"])
        return df.set_index("player", drop=False).sort_index()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Player stats file '{player_stats_file}' not found. Please check the file path."
        )
    except Exception as e:
        raise ValueError(f"Error occurred while loading player stats: {e}")


def query_data(df: pd.DataFrame, p1_name: str, p2_name: str, date: str) -> pd.DataFrame:
    """
    Query the historical data by player names and date.

    This function filters the DataFrame to find rows where the specified players
    (in any order) played on the given date.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing historical match data.
    p1_name : str
        The name of the first player.
    p2_name : str
        The name of the second player.
    date : str
        The date of the match in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the matching historical rows.

    Raises
    ------
    KeyError
        If the required columns ('p1', 'p2', 'date') are not present.
    ValueError
        If no match is found or an error occurs during querying.
    """
    try:
        date = pd.to_datetime(date)
        date_series = pd.to_datetime(df["date"])

        player_order = (df["p1"].eq(p1_name) & df["p2"].eq(p2_name)) | (
            df["p1"].eq(p2_name) & df["p2"].eq(p1_name)
        )
        matches = df[player_order & date_series.eq(date)]
        if matches.empty:
            raise ValueError(
                f"No match found for '{p1_name}' vs '{p2_name}' on '{date:%Y-%m-%d}'."
            )
        return matches
    except KeyError as e:
        raise KeyError(f"Invalid column names in the data: {e}")
    except Exception as e:
        raise ValueError(f"Error occurred while querying data: {e}")


def align_features_to_model(model: object, features: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the feature dataframe matches the model's expected column order.

    Parameters
    ----------
    model : Pipeline or BaseEstimator
        The trained Scikit-Learn model or pipeline.
    features : pd.DataFrame
        The assembled feature vector.

    Returns
    -------
    pd.DataFrame
        Features sorted and padded to match model training expectations.
    """
    # Extract expected feature names if available in the model/pipeline metadata
    expected_columns = getattr(model, "feature_names_in_", None)
    if expected_columns is None and hasattr(model, "named_steps"):
        encoder = model.named_steps.get("encoder")
        expected_columns = getattr(encoder, "feature_names_in_", None)

    if expected_columns is None:
        return features

    # Reorder columns and add missing ones as NA
    aligned = features.copy()
    for column in expected_columns:
        if column not in aligned.columns:
            aligned[column] = pd.NA
    return aligned.loc[:, list(expected_columns)]


def predict(model: object, df: pd.DataFrame):
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
    prob : float
        The probability of player 1 winning.
    class_ : int
        The binary class prediction (0 or 1).
    player_1 : str
        The name of player 1.
    """
    # 1. Select only columns allowed at inference time
    # 2. Align them to the specific order the model saw during training
    X = align_features_to_model(model, select_model_features(df))
    try:
        # Generate raw predictions
        prob = model.predict_proba(X)[:, 1]
        class_ = model.predict(X)
        return prob, class_, X["p1"].values[0]
    except Exception as e:
        raise ValueError(f"Error occurred during prediction: {e}")


def load_model(model_path: str | Path):
    """
    Load a trained model artifact.

    Parameters
    ----------
    model_path : str or Path
        Directory containing 'model_*.joblib' files, or a direct file path.

    Returns
    -------
    model : object
        The deserialized model artifact.
    """
    path = Path(model_path)
    if path.is_file():
        return load(path)

    # Automatically resolve the latest model in the directory by name
    model_files = [file for file in path.glob("model_*.joblib")]
    if not model_files:
        raise FileNotFoundError(
            f"No trained model found in '{model_path}'. Expected a 'model_*.joblib' file."
        )
    most_recent_model_file = max(model_files, key=lambda file: file.name)
    return load(most_recent_model_file)


def make_prediction(model: object, data: pd.DataFrame, p1_name: str, p2_name: str, date: str):
    """
    Score a match by looking it up in a historical dataframe.

    This helper is kept for regression checks against known historical rows.

    Parameters
    ----------
    model : object
        Pre-loaded model artifact.
    data : pd.DataFrame
        Pre-loaded historical match data.
    p1_name : str
        Name of player 1.
    p2_name : str
        Name of player 2.
    date : str
        Match date.

    Returns
    -------
    tuple
        (probability, class, player_1_name)
    """
    df_filtered = query_data(data, p1_name, p2_name, date)
    prob, class_, player_1 = predict(model, df_filtered)
    return prob, class_, player_1


def make_online_prediction(
    model: object,
    player_stats: pd.DataFrame,
    p1_name: str,
    p2_name: str,
    date: str,
    match_context: dict | None = None,
):
    """
    Assemble features from player state and match context, then score them.

    This is the "stateless" prediction path. It doesn't need the historical
    match to exist in the database; it builds the feature vector from
    the latest known player stats.

    Parameters
    ----------
    model : object
        Pre-loaded model artifact.
    player_stats : pd.DataFrame
        Pre-loaded 'latest_player_stats' table.
    p1_name : str
        Name of player 1.
    p2_name : str
        Name of player 2.
    date : str
        Match date.
    match_context : dict, optional
        Additional attributes like 'surface' or 'round'.

    Returns
    -------
    tuple
        (probability, class, player_1_name)
    """
    # Build the feature vector from scratch using current player state
    feature_row = build_match_feature_row(
        p1_name=p1_name,
        p2_name=p2_name,
        date=date,
        stats_context=player_stats,
        match_context=match_context,
    )
    # Run the standard prediction pipeline
    prob, class_, player_1 = predict(model, feature_row)
    return prob, class_, player_1
