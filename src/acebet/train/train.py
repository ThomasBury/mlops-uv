"""Training pipeline utilities for the AceBet classifier."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def prepare_data_for_training_clf(
    start_date: str, end_date: str
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load production data and build feature/target matrices for training.

    Args:
        start_date: Inclusive lower bound for match date filtering.
        end_date: Inclusive upper bound for match date filtering.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: Feature matrix and binary target vector.
    """
    data_path = (
        Path(__file__).resolve().parents[3] / "data" / "atp_data_production.feather"
    )
    df = pd.read_feather(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.query("date >= @start_date and date <= @end_date")

    predictors = df.columns.drop(
        ["target", "date", "sets_p1", "sets_p2", "b365_p1", "b365_p2", "ps_p1", "ps_p2"]
    )
    X = df[predictors].copy()
    y = df["target"].values.copy() * 1

    return X, y


def time_series_split(
    X: pd.DataFrame, y: np.ndarray, n_splits: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Create train/test indices using ``TimeSeriesSplit``.

    Args:
        X: Feature matrix sorted by time.
        y: Target array aligned with ``X``.
        n_splits: Number of time-based splits to generate.

    Returns:
        tuple[np.ndarray, np.ndarray]: Train and test index arrays from first split.
    """
    ts_split = TimeSeriesSplit(n_splits=n_splits)
    all_splits = list(ts_split.split(X, y))
    train_idx, test_idx = all_splits[0]
    return train_idx, test_idx


def train_model(start_date: str, end_date: str) -> None:
    """Train and serialize a LightGBM pipeline for winner prediction.

    Args:
        start_date: Inclusive lower bound for match date filtering.
        end_date: Inclusive upper bound for match date filtering.
    """
    X, y = prepare_data_for_training_clf(start_date, end_date)
    train_idx, _ = time_series_split(X, y, n_splits=2)
    X_train, y_train = X.iloc[train_idx, :].copy(), y[train_idx].copy()

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "feature_pre_filter": False,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "num_leaves": 4,
        "colsample_bytree": 0.4,
        "subsample": 0.7957346694832138,
        "subsample_freq": 4,
        "min_child_samples": 20,
        "n_estimators": 45,
    }

    model = Pipeline(
        [
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=np.nan
                ).set_output(transform="pandas"),
            ),
            ("gbm", LGBMClassifier(**lgb_params)),
        ]
    )

    model.fit(X_train, y_train)
    today = datetime.today()
    filename = f"./model_{today.strftime('%Y-%m-%d-%H-%M')}.joblib"
    dump(model, filename)


if __name__ == "__main__":
    train_model("2015-03-04", "2017-03-04")
