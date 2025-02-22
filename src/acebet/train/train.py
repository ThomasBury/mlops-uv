import pandas as pd
import numpy as np
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder
from joblib import dump
from datetime import datetime


def prepare_data_for_training_clf(start_date, end_date):
    """
    Prepare the ATP data for modeling.

    Parameters
    ----------
    start_date : str
        The start date of the time window.
    end_date : str
        The end date of the time window.

    Returns
    -------
    df : pandas.DataFrame
        The prepared data.

    """
    # replace by your production DB connection and table
    data_path = (
        Path(__file__).resolve().parents[3] / "data" / "atp_data_production.feather"
    )
    df = pd.read_feather(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.query("date >= @start_date and date <= @end_date")

    # Create predictors list
    predictors = df.columns.drop(
        ["target", "date", "sets_p1", "sets_p2", "b365_p1", "b365_p2", "ps_p1", "ps_p2"]
    )
    X = df[predictors].copy()
    y = df["target"].values.copy() * 1

    return X, y


def time_series_split(X, y, n_splits=2):
    """
    Split the data into training and test sets using a TimeSeriesSplit object.

    Parameters
    ----------
    X : pandas.DataFrame
        The features.
    y : pandas.Series
        The target.
    n_splits : int, default=2
        The number of splits.

    Returns
    -------
    train_idx : list
        The training indices.
    test_idx : list
        The test indices.

    """

    ts_split = TimeSeriesSplit(n_splits=n_splits)
    all_splits = list(ts_split.split(X, y))

    train_idx, test_idx = all_splits[0]
    return train_idx, test_idx


def train_model(start_date, end_date):
    """
    Train a model on the training data.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The training features.
    y_train : pandas.Series
        The training target.

    Returns
    -------
    model : sklearn.base.BaseEstimator
        The trained model.

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
    start_date = "2015-03-04"
    end_date = "2017-03-04"
    train_model(start_date, end_date)
