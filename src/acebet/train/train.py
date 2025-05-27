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
    # Load data from a feather file.
    # In a production environment, this might be replaced with a database connection.
    data_path = (
        Path(__file__).resolve().parents[3] / "data" / "atp_data_production.feather"
    )
    df = pd.read_feather(data_path) # Read data into a pandas DataFrame
    df["date"] = pd.to_datetime(df["date"]) # Convert 'date' column to datetime objects
    # Filter data to include only records within the specified date range.
    df = df.query("date >= @start_date and date <= @end_date")

    # Create predictors list by excluding target and other non-feature columns.
    # 'sets_p1', 'sets_p2' are results of the match and would cause data leakage.
    # Betting odds ('b365_p1', 'b365_p2', 'ps_p1', 'ps_p2') are also excluded as they might not be available at prediction time or introduce bias.
    predictors = df.columns.drop(
        ["target", "date", "sets_p1", "sets_p2", "b365_p1", "b365_p2", "ps_p1", "ps_p2"]
    )
    X = df[predictors].copy() # Create feature matrix X
    y = df["target"].values.copy() * 1 # Create target vector y, converting boolean to int (0 or 1)

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

    # LightGBM parameters for the classifier.
    # These parameters are chosen to control model complexity, prevent overfitting, and optimize performance.
    lgb_params = {
        "objective": "binary",  # Specifies the learning task as binary classification.
        "metric": "binary_logloss",  # Metric to be evaluated during training.
        "verbosity": -1,  # Controls the level of LightGBM's verbosity. -1 means silent.
        "boosting_type": "gbdt",  # Gradient Boosting Decision Tree.
        "feature_pre_filter": False, # Disable feature pre-filtering.
        "reg_alpha": 0.0,  # L1 regularization term on weights.
        "reg_lambda": 0.0,  # L2 regularization term on weights.
        "num_leaves": 4,  # Maximum number of leaves in one tree. Small value to reduce overfitting.
        "colsample_bytree": 0.4,  # Subsample ratio of columns when constructing each tree.
        "subsample": 0.7957346694832138,  # Subsample ratio of the training instance.
        "subsample_freq": 4,  # Frequency for subsampling.
        "min_child_samples": 20,  # Minimum number of data needed in a child (leaf).
        "n_estimators": 45,  # Number of boosting trees to be built.
    }

    # Create a scikit-learn pipeline to chain data preprocessing and modeling steps.
    model = Pipeline(
        [
            (
                "encoder", # Step 1: OrdinalEncoder
                OrdinalEncoder( # Encodes categorical features into an ordinal representation.
                    handle_unknown="use_encoded_value", unknown_value=np.nan # Handles unknown categories by assigning a specific value.
                ).set_output(transform="pandas"), # Ensures the output is a pandas DataFrame.
            ),
            ("gbm", LGBMClassifier(**lgb_params)), # Step 2: LightGBM Classifier with specified parameters.
        ]
    )

    model.fit(X_train, y_train) # Train the model pipeline on the training data.
    today = datetime.today()
    # Save the trained model to a file with a timestamp.
    filename = f"./model_{today.strftime('%Y-%m-%d-%H-%M')}.joblib"
    dump(model, filename) # Use joblib to serialize and save the model.


if __name__ == "__main__":
    start_date = "2015-03-04"
    end_date = "2017-03-04"
    train_model(start_date, end_date)
