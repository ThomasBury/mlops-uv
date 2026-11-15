"""Train, compare, and export the selected AceBet predictor."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from acebet import __version__
from acebet.dataprep.dataprep import prepare_data
from acebet.features import (
    CATEGORICAL_FEATURE_COLUMNS,
    FeatureContractPipeline,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    ordinal_codes_as_categories,
    select_model_features,
)
from acebet.model_artifacts import (
    MODEL_METADATA_SCHEMA_VERSION,
    candidates_with_selection_details,
    compute_selection,
    model_card_path,
    model_metadata_path,
    sha256_file,
    write_json,
)

from acebet.train.estimators import (
    EloBaselineClassifier,
    _require_binary_target_classes,
)

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "atp_data_production.feather"
DEFAULT_START_DATE = "2015-03-04"
DEFAULT_END_DATE = "2017-03-04"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "acebet"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"
DEFAULT_RANDOM_SEED = 42
MIN_TEMPORAL_HOLDOUT_ROWS = 4
MODEL_KIND_ELO = "elo_baseline"
MODEL_KIND_LIGHTGBM = "lightgbm"


@dataclass(frozen=True)
class PreparedTrainingData:
    """Feature, target, and calendar-date vectors for one training window."""

    features: pd.DataFrame
    target: np.ndarray
    dates: pd.Series


@dataclass(frozen=True)
class DateGroupedTemporalHoldout:
    """Indices and cutoff for one date-grouped temporal holdout."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    cutoff_date: pd.Timestamp


@dataclass
class TrainingResult:
    """Outputs and identities from one candidate-comparison training run."""

    exported_model_path: Path | None
    metrics: dict[str, float]
    mlflow_run_id: str | None
    candidate_metrics: dict[str, dict[str, float]]
    selection: dict[str, Any]
    model_kind: str
    package_version: str
    model_checksum: str | None = None


def get_lgb_params() -> dict[str, int | float | str | bool]:
    """Return deterministic hyperparameters for the LightGBM classifier."""
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
        "random_state": DEFAULT_RANDOM_SEED,
        "seed": DEFAULT_RANDOM_SEED,
        "feature_fraction_seed": DEFAULT_RANDOM_SEED,
        "bagging_seed": DEFAULT_RANDOM_SEED,
        "data_random_seed": DEFAULT_RANDOM_SEED,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 1,
    }


def prepare_data_for_training_clf(
    start_date: str, end_date: str
) -> PreparedTrainingData:
    """Load, validate, and inclusively slice the canonical ATP training window."""
    try:
        parsed_start_value = pd.Timestamp(start_date)
        if not isinstance(parsed_start_value, pd.Timestamp):
            raise ValueError
        parsed_start = parsed_start_value.normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Training window start_date must be a parseable date, got {start_date!r}."
        ) from exc
    try:
        parsed_end_value = pd.Timestamp(end_date)
        if not isinstance(parsed_end_value, pd.Timestamp):
            raise ValueError
        parsed_end = parsed_end_value.normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Training window end_date must be a parseable date, got {end_date!r}."
        ) from exc
    if parsed_start > parsed_end:
        raise ValueError(
            f"Training window is reversed: start_date {parsed_start.date()} must be on or before end_date {parsed_end.date()}."
        )

    prepared = prepare_data(DEFAULT_DATA_PATH)
    window = prepared.loc[
        prepared["date"].between(parsed_start, parsed_end, inclusive="both")
    ].reset_index(drop=True)
    if window.empty:
        raise ValueError(
            f"Training window contains no rows between {parsed_start.date()} and {parsed_end.date()} inclusive."
        )
    if len(window) < MIN_TEMPORAL_HOLDOUT_ROWS:
        raise ValueError(
            f"Training window is too small for a date-grouped temporal holdout: found {len(window)} rows, require at least {MIN_TEMPORAL_HOLDOUT_ROWS}."
        )
    dates = window["date"].copy()
    if dates.isna().any():
        raise ValueError("Training window dates must not contain null values.")
    if not dates.is_monotonic_increasing:
        raise ValueError(
            "Training window dates must be monotonically increasing before splitting."
        )
    if dates.nunique() < 2:
        raise ValueError(
            "Training window is too small for a date-grouped temporal holdout: at least two distinct calendar dates are required."
        )
    target = window["target"].to_numpy(dtype="int64", copy=True)
    _require_binary_target_classes(target, "Training window")
    features = select_model_features(window).reset_index(drop=True)
    if not (len(features) == len(target) == len(dates)):
        raise ValueError(
            "Prepared training data is misaligned: features, target, and dates must contain the same number of rows."
        )
    return PreparedTrainingData(features=features, target=target, dates=dates)


def date_grouped_temporal_holdout(
    data: PreparedTrainingData,
) -> DateGroupedTemporalHoldout:
    """Split once in time without placing one calendar date on both sides."""
    counts = (len(data.features), len(data.target), len(data.dates))
    if len(set(counts)) != 1:
        raise ValueError(
            f"Prepared training data is misaligned: features, target, and dates must contain the same number of rows; got {counts[0]}, {counts[1]}, and {counts[2]}."
        )
    if counts[2] < MIN_TEMPORAL_HOLDOUT_ROWS:
        raise ValueError(
            f"Training window is too small for a date-grouped temporal holdout: found {counts[2]} rows, require at least {MIN_TEMPORAL_HOLDOUT_ROWS}."
        )
    parsed = pd.to_datetime(data.dates, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Temporal holdout dates must not contain null values.")
    dates = parsed.dt.normalize().reset_index(drop=True)
    if not dates.is_monotonic_increasing:
        raise ValueError(
            "Temporal holdout dates must be monotonically increasing; sort preparation data stably before splitting."
        )
    if dates.nunique() < 2:
        raise ValueError(
            "Training window is too small for a date-grouped temporal holdout: at least two distinct calendar dates are required."
        )
    target = np.asarray(data.target)
    _require_binary_target_classes(target, "Training window")
    cutoff = pd.Timestamp(dates.iloc[(2 * len(dates)) // 3])
    if not isinstance(cutoff, pd.Timestamp):
        raise ValueError("Temporal holdout cutoff must be a valid date.")
    train_indices = np.flatnonzero(dates.lt(cutoff).to_numpy())
    test_indices = np.flatnonzero(dates.ge(cutoff).to_numpy())
    if train_indices.size == 0:
        raise ValueError(
            f"Date-grouped temporal holdout produced an empty training partition at cutoff {cutoff.date()}; choose a wider window with earlier dates."
        )
    if test_indices.size == 0:
        raise ValueError(
            f"Date-grouped temporal holdout produced an empty testing partition at cutoff {cutoff.date()}; choose a wider window with later dates."
        )
    _require_binary_target_classes(target[train_indices], "Training partition")
    _require_binary_target_classes(target[test_indices], "Testing partition")
    if dates.iloc[train_indices].max() >= dates.iloc[test_indices].min():
        raise ValueError("Date-grouped temporal holdout boundary is invalid.")
    return DateGroupedTemporalHoldout(train_indices, test_indices, cutoff)


def build_model(lgb_params: dict[str, Any]) -> FeatureContractPipeline:
    """Construct the deterministic feature-contract and LightGBM pipeline."""
    categorical = Pipeline(
        [
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    dtype=np.int32,
                ),
            ),
            (
                "as_category",
                FunctionTransformer(
                    ordinal_codes_as_categories, feature_names_out="one-to-one"
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", list(NUMERIC_FEATURE_COLUMNS)),
            ("categorical", categorical, list(CATEGORICAL_FEATURE_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return FeatureContractPipeline(
        [("preprocessor", preprocessor), ("gbm", LGBMClassifier(**lgb_params))]
    )


def evaluate_probabilities(y_test: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Evaluate binary probabilities using the four shared candidate metrics."""
    probability = np.asarray(y_proba, dtype=float)
    if probability.ndim != 1 or len(probability) != len(y_test):
        raise ValueError(
            "Evaluation probabilities must be one-dimensional and aligned with y_test."
        )
    if (
        not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("Evaluation probabilities must be finite and in [0, 1].")
    prediction = (probability > 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "log_loss": float(log_loss(y_test, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_test, probability)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
    }


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: np.ndarray
) -> dict[str, float]:
    """Evaluate a fitted probabilistic classifier using the shared evaluator."""
    return evaluate_probabilities(y_test, model.predict_proba(X_test)[:, 1])


def export_model(
    model: Any, output_dir: Path, exported_at: datetime | None = None
) -> Path:
    """Serialize a selected trusted model to a UTC timestamped joblib file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = exported_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    path = output_dir / f"model_{timestamp.strftime('%Y-%m-%d-%H-%M-%S')}.joblib"
    try:
        with path.open("xb") as artifact:
            dump(model, artifact)
    except FileExistsError as exc:
        raise FileExistsError(f"Model artifact already exists: {path}") from exc
    return path


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _date_range(dates: pd.Series) -> dict[str, str]:
    start = pd.Timestamp(dates.min())
    end = pd.Timestamp(dates.max())
    if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp):
        raise ValueError("Date range requires finite calendar dates.")
    return {
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
    }


def build_model_metadata(
    *,
    exported_model_path: Path,
    model_kind: str,
    start_date: str,
    end_date: str,
    data: PreparedTrainingData,
    holdout: DateGroupedTemporalHoldout,
    candidates: dict[str, Any],
    selection: dict[str, Any],
    mlflow_run_id: str | None,
    exported_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the complete machine-readable schema-v2 metadata payload."""
    train_dates = data.dates.iloc[holdout.train_indices]
    test_dates = data.dates.iloc[holdout.test_indices]
    timestamp = (exported_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        "schema_version": MODEL_METADATA_SCHEMA_VERSION,
        "dataset": {
            "source_path": _project_relative(DEFAULT_DATA_PATH),
            "source_sha256": sha256_file(DEFAULT_DATA_PATH),
            "requested_window": {"start_date": start_date, "end_date": end_date},
            "observed_window": _date_range(data.dates),
            "holdout_cutoff": holdout.cutoff_date.date().isoformat(),
            "train_date_range": _date_range(train_dates),
            "test_date_range": _date_range(test_dates),
            "row_counts": {
                "total": len(data.features),
                "train": len(holdout.train_indices),
                "test": len(holdout.test_indices),
            },
        },
        "feature_contract": {
            "columns": list(MODEL_FEATURE_COLUMNS),
            "numeric_columns": list(NUMERIC_FEATURE_COLUMNS),
            "categorical_columns": list(CATEGORICAL_FEATURE_COLUMNS),
            "count": len(MODEL_FEATURE_COLUMNS),
        },
        "candidates": candidates,
        "selection": selection,
        "model": {
            "filename": exported_model_path.name,
            "kind": model_kind,
            "package": {"name": "acebet", "version": __version__},
            "serialization": "joblib",
            "trusted_only": True,
            "sha256": sha256_file(exported_model_path),
        },
        "mlflow_run_id": mlflow_run_id,
        "exported_at_utc": timestamp.isoformat(),
    }


def write_model_metadata(exported_model_path: Path, metadata: dict[str, Any]) -> Path:
    """Write the JSON metadata sidecar."""
    return write_json(model_metadata_path(exported_model_path), metadata)


def write_model_card(exported_model_path: Path, metadata: dict[str, Any]) -> Path:
    """Write a concise human-readable card for the selected v2 artifact."""
    candidates, selection, dataset, identity = (
        metadata["candidates"],
        metadata["selection"],
        metadata["dataset"],
        metadata["model"],
    )
    lines = [
        "# AceBet Model Card",
        "",
        "## Model description",
        "",
        "AceBet predicts a tennis match winner from the strict prepared ATP feature contract.",
        "",
        "## Intended uses & limitations",
        "",
        "This is a tutorial artifact and is not suitable for betting or other high-stakes decisions.",
        "",
        "## Candidate comparison",
        "",
        "| Candidate | Accuracy | Log loss | Brier score | ROC AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("constant", "elo", "lightgbm"):
        metric = candidates[name]["metrics"]
        lines.append(
            f"| {name} | {metric['accuracy']:.5f} | {metric['log_loss']:.5f} | {metric['brier_score']:.5f} | {metric['roc_auc']:.5f} |"
        )
    lines += [
        "",
        "## Evaluation Results",
        "",
        "Learned candidates must improve unrounded log loss over constant by at least 0.005. "
        "LightGBM replaces a servable Elo model only when it improves unrounded log loss by at least 0.005.",
        "",
        f"- Selected model kind: `{identity['kind']}`",
        f"- Selection reason: `{selection['reason_code']}` — {selection['reason']}",
        f"- Dataset SHA-256: `{dataset['source_sha256']}`",
        f"- Model SHA-256: `{identity['sha256']}`",
        f"- Model file: `{identity['filename']}`",
        f"- Package identity: `acebet=={identity['package']['version']}`",
        "",
        "## Security warning",
        "",
        "Checksums provide integrity, not authenticity. Joblib uses pickle and this artifact must only be loaded from a trusted source.",
        "",
    ]
    output = model_card_path(exported_model_path)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _import_mlflow() -> tuple[Any, Any]:
    try:
        mlflow = import_module("mlflow")
        mlflow_models = import_module("mlflow.models")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLflow support requires the optional 'mlops' dependency group. Install it with `uv sync --group mlops`."
        ) from exc
    return mlflow, getattr(mlflow_models, "infer_signature")


def configure_mlflow(
    mlflow: Any, tracking_uri: str, experiment_name: str, artifact_root: Path
) -> None:
    """Configure local or remote MLflow tracking."""
    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri.startswith(("http://", "https://")):
        mlflow.set_experiment(experiment_name)
        return
    artifact_root.mkdir(parents=True, exist_ok=True)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name, artifact_location=artifact_root.resolve().as_uri()
        )
    mlflow.set_experiment(experiment_name)


def log_training_run(
    *,
    model: Any | None,
    model_kind: str | None,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    candidates: dict[str, Any],
    selection: dict[str, Any],
    start_date: str,
    end_date: str,
    tracking_uri: str,
    experiment_name: str,
    artifact_root: Path,
) -> str:
    """Log every comparison and the selected sklearn model when one exists."""
    mlflow, infer_signature = _import_mlflow()
    configure_mlflow(mlflow, tracking_uri, experiment_name, artifact_root)
    metrics = {
        f"candidate.{name}.{metric}": value
        for name in ("constant", "elo", "lightgbm")
        for metric, value in candidates[name]["metrics"].items()
    }
    metrics.update(
        {f"selection.{name}": value for name, value in selection["deltas"].items()}
    )
    params = {
        "start_date": start_date,
        "end_date": end_date,
        "selection.learned_vs_constant_margin": selection["learned_vs_constant_margin"],
        "selection.lightgbm_vs_elo_margin": selection["lightgbm_vs_elo_margin"],
        "selection.status": selection["status"],
        "selection.selected_model_kind": model_kind or "none",
        "selection.reason_code": selection["reason_code"],
    }
    params.update(
        {
            f"lightgbm.{key}": value
            for key, value in candidates["lightgbm"]["parameters"].items()
        }
    )
    with mlflow.start_run() as run:
        mlflow.set_tags(
            {
                "project": "acebet",
                "selection_reason": selection["reason_code"],
                "selected_model_kind": model_kind or "none",
            }
        )
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if model is not None:
            example = X_test.head(5) if not X_test.empty else X_train.head(5)
            signature = infer_signature(example, model.predict_proba(example))
            mlflow.sklearn.log_model(
                sk_model=model, name="model", signature=signature, input_example=example
            )
        return run.info.run_id


def _log_bundle_to_run(
    run_id: str, exported_model_path: Path, tracking_uri: str
) -> None:
    mlflow, _ = _import_mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=run_id):
        for path in (
            exported_model_path,
            model_metadata_path(exported_model_path),
            model_card_path(exported_model_path),
        ):
            mlflow.log_artifact(str(path), artifact_path="exports")


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
    """Fit all candidates, select by log loss, and export only the selected model."""
    data = prepare_data_for_training_clf(start_date, end_date)
    holdout = date_grouped_temporal_holdout(data)
    X_train = data.features.iloc[holdout.train_indices, :].copy()
    y_train = data.target[holdout.train_indices].copy()
    X_test = data.features.iloc[holdout.test_indices, :].copy()
    y_test = data.target[holdout.test_indices].copy()

    prevalence = float(np.mean(y_train))
    elo_model = EloBaselineClassifier().fit(X_train, y_train)
    lgb_params = get_lgb_params()
    lightgbm_model = build_model(lgb_params).fit(X_train, y_train)
    candidate_metrics = {
        "constant": evaluate_probabilities(y_test, np.full(len(y_test), prevalence)),
        "elo": evaluate_model(elo_model, X_test, y_test),
        "lightgbm": evaluate_model(lightgbm_model, X_test, y_test),
    }
    candidates = candidates_with_selection_details(
        candidate_metrics, prevalence, lgb_params
    )
    selection = compute_selection(candidates)
    selected_kind = selection["selected_model_kind"]
    selected_model = {
        MODEL_KIND_ELO: elo_model,
        MODEL_KIND_LIGHTGBM: lightgbm_model,
    }.get(selected_kind)

    mlflow_run_id = None
    if enable_mlflow:
        mlflow_run_id = log_training_run(
            model=selected_model,
            model_kind=selected_kind,
            X_train=X_train,
            X_test=X_test,
            candidates=candidates,
            selection=selection,
            start_date=start_date,
            end_date=end_date,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            artifact_root=artifact_root,
        )
    if selected_model is None:
        LOGGER.error("Model selection failed: %s", selection["reason_code"])
        raise ValueError(
            f"Model selection failed: {selection['reason_code']}. {selection['reason']}"
        )

    exported_path = None
    checksum = None
    if export_joblib:
        exported_at = datetime.now(timezone.utc)
        exported_path = export_model(
            selected_model, output_dir, exported_at=exported_at
        )
        try:
            metadata = build_model_metadata(
                exported_model_path=exported_path,
                model_kind=selected_kind,
                start_date=start_date,
                end_date=end_date,
                data=data,
                holdout=holdout,
                candidates=candidates,
                selection=selection,
                mlflow_run_id=mlflow_run_id,
                exported_at=exported_at,
            )
            write_model_metadata(exported_path, metadata)
            write_model_card(exported_path, metadata)
            checksum = metadata["model"]["sha256"]
        except Exception:
            for path in (
                exported_path,
                model_metadata_path(exported_path),
                model_card_path(exported_path),
            ):
                path.unlink(missing_ok=True)
            raise
        if mlflow_run_id is not None:
            _log_bundle_to_run(mlflow_run_id, exported_path, tracking_uri)

    return TrainingResult(
        exported_model_path=exported_path,
        metrics=dict(
            candidate_metrics["elo" if selected_kind == MODEL_KIND_ELO else "lightgbm"]
        ),
        mlflow_run_id=mlflow_run_id,
        candidate_metrics=candidate_metrics,
        selection=selection,
        model_kind=selected_kind,
        package_version=__version__,
        model_checksum=checksum,
    )


def parse_args() -> argparse.Namespace:
    """Parse the fixed-policy training CLI."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train and select the AceBet match winner model.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        metavar="YYYY-MM-DD",
        help="Start of the training/selection window (inclusive).",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        metavar="YYYY-MM-DD",
        help="End of the training/selection window (inclusive).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where an optional selected v2 bundle is written.",
    )
    parser.add_argument(
        "--export-joblib",
        action="store_true",
        help="Export only the selected servable predictor.",
    )
    parser.add_argument(
        "--enable-mlflow",
        action="store_true",
        help="Log all candidate comparisons and the selected model to MLflow.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI when tracking is enabled.",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment when tracking is enabled.",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="Local artifact root for a new MLflow experiment.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
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
    print(f"Selected model: {result.model_kind} ({result.selection['reason_code']})")
    for name, metrics in result.candidate_metrics.items():
        print(
            f"{name}: log_loss={metrics['log_loss']}, accuracy={metrics['accuracy']}, brier_score={metrics['brier_score']}, roc_auc={metrics['roc_auc']}"
        )
    if result.exported_model_path is not None:
        print(f"Exported model: {result.exported_model_path}")
    if result.mlflow_run_id is not None:
        print(f"MLflow run ID: {result.mlflow_run_id}")


if __name__ == "__main__":
    main()
