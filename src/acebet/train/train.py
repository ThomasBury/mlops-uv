import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from acebet.features import select_model_features
from acebet.dataprep.dataprep import prepare_data

# Define project-wide defaults
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "atp_data_production.feather"
DEFAULT_START_DATE = "2015-03-04"
DEFAULT_END_DATE = "2017-03-04"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "acebet"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"


@dataclass
class TrainingResult:
    """
    Data container for model training outputs.

    Attributes
    ----------
    exported_model_path : Path or None
        Location where the joblib artifact was saved.
    metrics : dict[str, float]
        Performance metrics (accuracy, log_loss).
    mlflow_run_id : str or None
        The ID of the tracked MLflow run.
    """
    exported_model_path: Path | None
    metrics: dict[str, float]
    mlflow_run_id: str | None


def get_lgb_params() -> dict[str, int | float | str | bool]:
    """
    Return static hyperparameters for the LightGBM classifier.
    """
    return {
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


def prepare_data_for_training_clf(start_date: str, end_date: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load, validate, and slice the ATP data for a specific training window.

    This function leverages the shared 'prepare_data' logic to ensure
    the training set undergoes the exact same feature engineering as
    production data.

    Parameters
    ----------
    start_date : str
        ISO date string for the start of the training window.
    end_date : str
        ISO date string for the end of the training window.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        (Features X, Labels y)
    """
    # 1. Standardize and engineer features using the dataprep module
    df = prepare_data(DEFAULT_DATA_PATH)
    
    # 2. Slice the dataset temporally
    df = df.query("date >= @start_date and date <= @end_date")

    # 3. Apply the shared feature contract to drop forbidden/leaky columns
    X = select_model_features(df)
    y = df["target"].values.copy() * 1

    return X, y


def time_series_split(
    X: pd.DataFrame, y: np.ndarray, n_splits: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the data into training and test indices using a TimeSeriesSplit.

    In MLOps for predictive tasks, we avoid random shuffling to prevent
    'data leakage' from the future into the past.

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix.
    y : np.ndarray
        The target vector.
    n_splits : int
        Number of temporal folds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (train_indices, test_indices) for the latest fold.
    """
    ts_split = TimeSeriesSplit(n_splits=n_splits)
    all_splits = list(ts_split.split(X, y))
    train_idx, test_idx = all_splits[-1]
    return train_idx, test_idx


def build_model(lgb_params: dict[str, Any]) -> Pipeline:
    """
    Construct the Scikit-Learn pipeline.

    The pipeline includes an OrdinalEncoder to handle categorical strings
    consistently across training and serving.
    """
    return Pipeline(
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


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, float]:
    """
    Calculate performance metrics on a hold-out test set.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, y_proba, labels=[0, 1])),
    }


def export_model(model: Pipeline, output_dir: Path) -> Path:
    """
    Serialize the trained model to a timestamped joblib file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"model_{datetime.today().strftime('%Y-%m-%d-%H-%M')}.joblib"
    dump(model, filename)
    return filename


def _import_mlflow() -> tuple[Any, Any]:
    """
    Lazy-load MLflow to avoid dependency errors if the group is not installed.
    """
    try:
        import mlflow
        from mlflow.models import infer_signature
    except ImportError as exc:
        raise RuntimeError(
            "MLflow support requires the optional 'mlops' dependency group. "
            "Install it with `uv sync --group mlops`."
        ) from exc
    return mlflow, infer_signature


def configure_mlflow(
    mlflow: Any,
    tracking_uri: str,
    experiment_name: str,
    artifact_root: Path,
) -> None:
    """
    Set up the MLflow tracking environment.
    """
    mlflow.set_tracking_uri(tracking_uri)

    if tracking_uri.startswith(("http://", "https://")):
        mlflow.set_experiment(experiment_name)
        return

    # Local SQLite configuration
    artifact_root.mkdir(parents=True, exist_ok=True)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name, artifact_location=artifact_root.resolve().as_uri()
        )
    mlflow.set_experiment(experiment_name)


def log_training_run(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    metrics: dict[str, float],
    lgb_params: dict[str, Any],
    start_date: str,
    end_date: str,
    tracking_uri: str,
    experiment_name: str,
    artifact_root: Path,
    exported_model_path: Path | None,
) -> str:
    """
    Log parameters, metrics, and artifacts to MLflow.
    """
    mlflow, infer_signature = _import_mlflow()
    configure_mlflow(mlflow, tracking_uri, experiment_name, artifact_root)

    # Automatically infer the input/output schema for the model
    input_example = X_test.head(5) if not X_test.empty else X_train.head(5)
    signature = infer_signature(input_example, model.predict_proba(input_example))

    run_params: dict[str, Any] = {
        "start_date": start_date,
        "end_date": end_date,
        "data_path": str(DEFAULT_DATA_PATH),
        "feature_count": int(X_train.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    run_params.update({f"lgbm.{key}": value for key, value in lgb_params.items()})

    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "project": "acebet",
                "tutorial": "mlops-uv",
                "framework": "lightgbm",
            }
        )
        mlflow.log_params(run_params)
        mlflow.log_metrics(metrics)
        # Log the pipeline itself as a first-class model object
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
        )
        if exported_model_path is not None:
            mlflow.log_artifact(str(exported_model_path), artifact_path="exports")
        return run.info.run_id


def train_model(
    start_date: str,
    end_date: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    export_joblib: bool = False,
    enable_mlflow: bool = False,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> TrainingResult:
    """
    High-level entry point for the training workflow.
    """
    # 1. Data Ingestion
    X, y = prepare_data_for_training_clf(start_date, end_date)
    
    # 2. Time-aware Splitting
    train_idx, test_idx = time_series_split(X, y, n_splits=2)
    X_train, y_train = X.iloc[train_idx, :].copy(), y[train_idx].copy()
    X_test, y_test = X.iloc[test_idx, :].copy(), y[test_idx].copy()

    # 3. Fit Model
    lgb_params = get_lgb_params()
    model = build_model(lgb_params)
    model.fit(X_train, y_train)

    # 4. Evaluation
    metrics = evaluate_model(model, X_test, y_test)
    
    # 5. Serialization
    exported_model_path = export_model(model, output_dir) if export_joblib else None
    
    # 6. Tracking
    mlflow_run_id = None
    if enable_mlflow:
        mlflow_run_id = log_training_run(
            model=model,
            X_train=X_train,
            X_test=X_test,
            metrics=metrics,
            lgb_params=lgb_params,
            start_date=start_date,
            end_date=end_date,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            artifact_root=artifact_root,
            exported_model_path=exported_model_path,
        )

    return TrainingResult(
        exported_model_path=exported_model_path,
        metrics=metrics,
        mlflow_run_id=mlflow_run_id,
    )


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for training.
    """
    parser = argparse.ArgumentParser(description="Train the AceBet match winner model.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where exported joblib files are written.",
    )
    parser.add_argument(
        "--export-joblib",
        action="store_true",
        help="Export the trained model as a timestamped joblib file.",
    )
    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        help="Log parameters, metrics, and the trained model to MLflow.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI, for example sqlite:///mlflow.db or http://127.0.0.1:5000.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name to use when logging runs.",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="Local artifact directory used when logging directly to SQLite.",
    )
    return parser.parse_args()


def main() -> None:
    """
    CLI script entry point.
    """
    args = parse_args()
    result = train_model(
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=Path(args.output_dir),
        export_joblib=args.export_joblib,
        enable_mlflow=args.enable_mlflow,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        artifact_root=Path(args.artifact_root),
    )

    print(f"Accuracy: {result.metrics['accuracy']:.4f}")
    print(f"Log loss: {result.metrics['log_loss']:.4f}")
    if result.exported_model_path is not None:
        print(f"Exported model: {result.exported_model_path}")
    if result.mlflow_run_id is not None:
        print(f"MLflow run ID: {result.mlflow_run_id}")


if __name__ == "__main__":
    main()
