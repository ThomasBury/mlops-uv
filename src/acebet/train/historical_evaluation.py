"""Compare, freeze, and score the tutorial's historical match forecasts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date, timedelta
from importlib.metadata import version
import json
from pathlib import Path
from platform import python_version
from tempfile import TemporaryDirectory
from typing import cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from acebet.dataprep.historical_replay import (
    DEVELOPMENT_END,
    F01_SURFACE,
    INITIAL_ELO,
    K,
    leaky_elo_probability,
    prior_day_cutoff,
    replay_matches,
    worked_fixture,
)
from acebet.dataprep.historical_results import (
    SNAPSHOT_SHA256,
    MatchResult,
    content_sha256,
    read_results,
)
from acebet.model_artifacts import read_json, sha256_file

VALIDATION_STARTS = (date(2021, 3, 1), date(2021, 5, 1), date(2021, 7, 1))
FEATURES = ["elo_gap", "surface"]
SEED = 42
C_VALUES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
SEARCH_TRIALS = 5
BOOTSTRAP_DRAWS = 2000
REPLAY_ATOL = 1e-8
CALIBRATION_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MEMBERSHIP = ["match_id", "event_date", "available_at", "cutoff", "p1_id", "p2_id"]


def replay_frame(results: Sequence[MatchResult], *, leak: bool = False) -> pd.DataFrame:
    """Arrange M3 replay outputs for fitting and inspection."""
    return pd.DataFrame.from_records(
        {
            "match_id": match.match_id,
            "event_date": match.event_date,
            "available_at": match.available_at,
            "cutoff": forecast.cutoff,
            "p1_id": match.p1_id,
            "p2_id": match.p2_id,
            "label": match.label,
            "surface": match.surface,
            "elo_gap": forecast.p1_elo - forecast.p2_elo,
            "elo": forecast.p1_win_probability,
            **(
                {"leaky_elo": leaky_elo_probability(forecast, match.label)}
                if leak
                else {}
            ),
        }
        for match, forecast in replay_matches(results)
    )


def development_results(results: Sequence[MatchResult]) -> list[MatchResult]:
    """Exclude labels that arrive after the model has to be frozen."""
    cutoff = prior_day_cutoff(DEVELOPMENT_END)
    return [
        row
        for row in results
        if row.event_date < DEVELOPMENT_END and row.available_at <= cutoff
    ]


def mature_indices(frame: pd.DataFrame, start: date) -> np.ndarray:
    """Only fit on matches whose labels were visible at the boundary forecast."""
    cutoff = prior_day_cutoff(start)
    return np.flatnonzero(
        (frame["event_date"] < start) & (frame["available_at"] <= cutoff)
    )


def development_folds(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding training windows and disjoint validation windows."""
    folds = []
    for start, end in zip(VALIDATION_STARTS, (*VALIDATION_STARTS[1:], DEVELOPMENT_END)):
        train = mature_indices(frame, start)
        valid = np.flatnonzero(
            frame["event_date"].between(start, end, inclusive="left")
        )
        if not len(valid) or frame.iloc[train]["label"].nunique() != 2:
            raise ValueError(
                f"Need both training labels and validation rows for {start}."
            )
        folds.append((train, valid))
    return folds


def logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        ("elo", StandardScaler(), ["elo_gap"]),
                        (
                            "surface",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["surface"],
                        ),
                    ]
                ),
            ),
            ("model", LogisticRegression(random_state=SEED)),
        ]
    )


def _prevalence(train: pd.DataFrame) -> float:
    return float(train["label"].mean())


def _comparison(
    frame: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[dict, pd.DataFrame]:
    fold_scores = []
    for train_idx, valid_idx in folds:
        train, valid = frame.iloc[train_idx], frame.iloc[valid_idx]
        model = logistic_pipeline().fit(train[FEATURES], train["label"])
        predictions = {
            "prevalence": np.full(len(valid), _prevalence(train)),
            "elo": valid["elo"],
            "logistic_default": model.predict_proba(valid[FEATURES])[:, 1],
            "leaky_elo": valid["leaky_elo"],
        }
        fold_scores.append(
            {
                name: log_loss(valid["label"], p, labels=[0, 1])
                for name, p in predictions.items()
            }
        )

    random_train, random_valid = train_test_split(
        frame, test_size=len(folds[-1][1]), random_state=SEED
    )
    random_model = logistic_pipeline().fit(
        random_train[FEATURES], random_train["label"]
    )
    latest = random_train["available_at"].max()
    report = {
        "fold_log_loss": fold_scores,
        "mean_log_loss": pd.DataFrame(fold_scores).mean().to_dict(),
        "random_split_diagnostic": {
            "log_loss": log_loss(
                random_valid["label"],
                random_model.predict_proba(random_valid[FEATURES]),
                labels=[0, 1],
            ),
            "validation_rows": len(random_valid),
            "validation_rows_with_later_training_labels": int(
                (random_valid["cutoff"] < latest).sum()
            ),
            "earliest_validation_cutoff": random_valid["cutoff"].min().isoformat(),
            "latest_training_label_available_at": latest.isoformat(),
        },
    }
    random_membership = pd.concat(
        [
            random_train[MEMBERSHIP].assign(partition="train"),
            random_valid[MEMBERSHIP].assign(partition="validation"),
        ]
    ).assign(split="random_diagnostic")
    return report, random_membership


def _fixture_predictions(model: Pipeline) -> dict:
    fixture = worked_fixture()
    gap = cast(float, fixture["p1_elo"]) - cast(float, fixture["p2_elo"])
    inputs = pd.DataFrame({"elo_gap": [gap, gap], "surface": [F01_SURFACE, "Unseen"]})
    known, unseen = model.predict_proba(inputs)[:, 1]
    encoder = model.named_steps["features"].named_transformers_["surface"]
    return {
        **fixture,
        "surface": F01_SURFACE,
        "logistic_p1_win_probability": float(known),
        "surfaces_seen_in_fit": encoder.categories_[0].tolist(),
        "unseen_surface": "Unseen",
        "unseen_surface_encoding": encoder.transform(inputs.iloc[[1]][["surface"]])
        .toarray()[0]
        .tolist(),
        "unseen_surface_p1_win_probability": float(unseen),
    }


def _runtime_identity() -> dict:
    package = Path(__file__).resolve().parents[1]
    sources = (
        "dataprep/historical_results.py",
        "dataprep/historical_replay.py",
        "features.py",
        "forecast.py",
        "model_artifacts.py",
        "train/historical_evaluation.py",
    )
    return {
        "sources_sha256": {name: sha256_file(package / name) for name in sources},
        "dependencies": {
            "python": python_version(),
            **{
                name: version(name)
                for name in (
                    "joblib",
                    "mlflow",
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "scikit-learn",
                    "scipy",
                    "threadpoolctl",
                )
            },
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _tracking_experiment(parent: Path):
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{parent.resolve() / 'mlflow.db'}")
    name = "acebet-historical-forecast"
    experiment = mlflow.get_experiment_by_name(name)
    return (
        experiment.experiment_id
        if experiment
        else mlflow.create_experiment(
            name, artifact_location=(parent / "mlartifacts").resolve().as_uri()
        )
    )


def train_candidate(results: Sequence[MatchResult], output: Path) -> dict:
    """Freeze the temporal search decision and fitted preprocessing before scoring."""
    import mlflow

    if output.exists():
        raise FileExistsError(output)
    development = development_results(results)
    frame = replay_frame(development, leak=True)
    folds = development_folds(frame)
    search = RandomizedSearchCV(
        logistic_pipeline(),
        {"model__C": C_VALUES},
        n_iter=SEARCH_TRIALS,
        scoring="neg_log_loss",
        cv=folds,
        random_state=SEED,
        refit=False,
        error_score="raise",
    ).fit(frame[FEATURES], frame["label"])
    model = (
        logistic_pipeline()
        .set_params(**search.best_params_)
        .fit(frame[FEATURES], frame["label"])
    )
    comparison, random_membership = _comparison(frame, folds)
    metrics = comparison["mean_log_loss"]
    metrics["logistic_tuned"] = -float(search.best_score_)
    membership = [frame[MEMBERSHIP].assign(split="final_refit", partition="train")]
    for number, (train_idx, valid_idx) in enumerate(folds, 1):
        for partition, indices in (("train", train_idx), ("validation", valid_idx)):
            membership.append(
                frame.iloc[indices][MEMBERSHIP].assign(
                    split=f"fold_{number}", partition=partition
                )
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / output.name
        bundle.mkdir()
        joblib.dump(model, bundle / "model.joblib")
        pd.concat([*membership, random_membership]).to_csv(
            bundle / "membership.csv", index=False
        )
        trials = pd.DataFrame(search.cv_results_).filter(
            regex=r"^(params$|split\d+_test_score$|mean_test_score$|rank_test_score$)"
        )
        trials.to_csv(bundle / "search.csv", index=False)
        with mlflow.start_run(
            experiment_id=_tracking_experiment(output.parent), run_name="selection"
        ) as run:
            record = {
                "run_id": run.info.run_id,
                "source_sha256": SNAPSHOT_SHA256,
                "data_sha256": content_sha256(results),
                "development_sha256": content_sha256(development),
                **_runtime_identity(),
                "config": {
                    "validation_starts": [day.isoformat() for day in VALIDATION_STARTS],
                    "final_start": DEVELOPMENT_END.isoformat(),
                    "selection_label_cutoff": prior_day_cutoff(
                        DEVELOPMENT_END
                    ).isoformat(),
                    "features": FEATURES,
                    "initial_elo": INITIAL_ELO,
                    "elo_k": K,
                    "c_values": C_VALUES,
                    "search_trials": SEARCH_TRIALS,
                    "seed": SEED,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_block": "nonempty UTC calendar week starting Monday",
                    "calibration_bin_edges": CALIBRATION_BINS,
                    "replay_atol": REPLAY_ATOL,
                    "replay_rtol": 0.0,
                    "selection_rule": "Lowest mean temporal fold log loss among sampled logistic settings, first sampled on an exact tie.",
                },
                "best_params": search.best_params_,
                "fitted_model_params": model.named_steps["model"].get_params(),
                "refit_prevalence": _prevalence(frame),
                **comparison,
                "fixture": _fixture_predictions(model),
                "artifacts_sha256": {
                    path.name: sha256_file(path) for path in sorted(bundle.iterdir())
                },
            }
            _write_json(bundle / "selection.json", record)
            mlflow.log_params(
                {
                    **record["best_params"],
                    "seed": SEED,
                    "data_sha256": record["data_sha256"],
                    "model_sha256": record["artifacts_sha256"]["model.joblib"],
                }
            )
            mlflow.log_metrics(
                {f"dev_{name}_log_loss": value for name, value in metrics.items()}
            )
            mlflow.log_artifacts(str(bundle))
        bundle.rename(output)
    return record


def _losses(label: np.ndarray, probability: np.ndarray) -> np.ndarray:
    eps = np.finfo(float).eps
    probability = probability.clip(eps, 1 - eps)
    return -np.log(np.where(label == 1, probability, 1 - probability))


def _calibration(label: np.ndarray, probability: np.ndarray) -> list[dict]:
    bins = np.searchsorted(CALIBRATION_BINS[1:-1], probability, side="right")
    summary = []
    for index, (left, right) in enumerate(zip(CALIBRATION_BINS, CALIBRATION_BINS[1:])):
        mask = bins == index
        count = int(mask.sum())
        summary.append(
            {
                "lower": left,
                "upper": right,
                "matches": count,
                "mean_probability": float(probability[mask].mean()) if count else None,
                "observed_win_rate": float(label[mask].mean()) if count else None,
            }
        )
    return summary


def paired_week_interval(delta: np.ndarray, days: pd.Series) -> dict:
    """Resample whole nonempty calendar weeks and preserve match-level pairing."""
    mondays = [day - timedelta(days=day.weekday()) for day in days]
    blocks = pd.Series(delta).groupby(mondays).agg(["sum", "size"])
    if len(blocks) < 2:
        raise ValueError("Paired uncertainty requires at least two nonempty weeks.")
    sampled = np.random.default_rng(SEED).integers(
        len(blocks), size=(BOOTSTRAP_DRAWS, len(blocks))
    )
    draws = blocks["sum"].to_numpy()[sampled].sum(axis=1) / blocks["size"].to_numpy()[
        sampled
    ].sum(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_delta": float(delta.mean()),
        "interval_95": [float(low), float(high)],
        "weeks": len(blocks),
    }


def evaluate_candidate(results: Sequence[MatchResult], bundle: Path) -> dict:
    """Score the frozen model once without fitting or tuning on final results."""
    import mlflow

    output = bundle / "final"
    if output.exists():
        raise FileExistsError(output)
    selection_path = bundle / "selection.json"
    selection_hash = sha256_file(selection_path)
    selection = read_json(selection_path)
    if selection["data_sha256"] != content_sha256(results):
        raise ValueError("Canonical results differ from the frozen selection.")
    if any(selection[key] != value for key, value in _runtime_identity().items()):
        raise ValueError("Code or dependencies differ from the frozen selection.")
    for name in ("model.joblib", "membership.csv", "search.csv"):
        if sha256_file(bundle / name) != selection["artifacts_sha256"][name]:
            raise ValueError(f"{name} differs from the frozen selection.")
    # Joblib is only for trusted bundles produced locally by the training command.
    model = joblib.load(bundle / "model.joblib")
    frame = replay_frame(results)
    final = frame.loc[frame["event_date"] >= DEVELOPMENT_END].reset_index(drop=True)
    if final.empty:
        raise ValueError("The canonical results contain no final fixtures.")
    predictions = {
        "prevalence": np.full(len(final), selection["refit_prevalence"]),
        "elo": final["elo"].to_numpy(),
        "logistic_tuned": model.predict_proba(final[FEATURES])[:, 1],
    }
    labels = final["label"].to_numpy()
    losses = {name: _losses(labels, p) for name, p in predictions.items()}
    metrics = {name: float(values.mean()) for name, values in losses.items()}
    comparisons = {
        baseline: paired_week_interval(
            losses["logistic_tuned"] - losses[baseline], final["event_date"]
        )
        for baseline in ("prevalence", "elo")
    }
    with TemporaryDirectory(dir=bundle) as temporary:
        artifacts = Path(temporary) / "final"
        artifacts.mkdir()
        final[MEMBERSHIP + FEATURES + ["label"]].assign(**predictions).to_csv(
            artifacts / "forecasts.csv", index=False
        )
        with mlflow.start_run(
            experiment_id=_tracking_experiment(bundle.parent),
            run_name="final",
            tags={"selection_run_id": selection["run_id"]},
        ) as run:
            report = {
                "run_id": run.info.run_id,
                "selection_run_id": selection["run_id"],
                "selection_sha256": selection_hash,
                "model_sha256": selection["artifacts_sha256"]["model.joblib"],
                "forecasts_sha256": sha256_file(artifacts / "forecasts.csv"),
                "matches": len(final),
                "log_loss": metrics,
                "calibration": {
                    name: _calibration(labels, p) for name, p in predictions.items()
                },
                "log_loss_delta_candidate_minus_baseline": comparisons,
            }
            _write_json(artifacts / "evaluation.json", report)
            mlflow.log_metrics(
                {f"final_{name}_log_loss": value for name, value in metrics.items()}
            )
            mlflow.log_artifacts(str(artifacts))
        artifacts.rename(output)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("compare", "train", "evaluate"):
        subparser = commands.add_parser(name)
        subparser.add_argument("results", type=Path)
        if name != "compare":
            subparser.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)
    results = read_results(args.results)
    if args.command == "compare":
        frame = replay_frame(development_results(results), leak=True)
        report, _ = _comparison(frame, development_folds(frame))
    elif args.command == "train":
        report = train_candidate(results, args.bundle)
    else:
        report = evaluate_candidate(results, args.bundle)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
