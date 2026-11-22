"""Compare, freeze, and score the tutorial's historical match forecasts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import nullcontext
from datetime import date, timedelta
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from platform import python_version
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
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

if TYPE_CHECKING:
    from optuna.trial import Trial

VALIDATION_STARTS = (date(2021, 3, 1), date(2021, 5, 1), date(2021, 7, 1))
FEATURES = ["elo_gap", "surface"]
SEED = 42
C_BOUNDS = (0.001, 10.0)
SEARCH_TRIALS = 12
STARTUP_TRIALS = 5
BOOTSTRAP_DRAWS = 2000
REPLAY_ATOL = 1e-8
CALIBRATION_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MEMBERSHIP = ["match_id", "event_date", "available_at", "cutoff", "p1_id", "p2_id"]


def replay_frame(results: Sequence[MatchResult], *, leak: bool = False) -> pd.DataFrame:
    """Arrange historical replay outputs for fitting and inspection."""
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


def _fold_membership(
    frame: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]
) -> list[pd.DataFrame]:
    return [
        frame.iloc[indices][MEMBERSHIP].assign(
            split=f"fold_{number}", partition=partition
        )
        for number, (train_idx, valid_idx) in enumerate(folds, 1)
        for partition, indices in (("train", train_idx), ("validation", valid_idx))
    ]


def logistic_pipeline(c: float = 1.0) -> Pipeline:
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
            ("model", LogisticRegression(C=c, random_state=SEED)),
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
    latest_train = random_train.loc[random_train["available_at"].idxmax()]
    earliest_valid = random_valid.loc[random_valid["cutoff"].idxmin()]
    latest = latest_train["available_at"]
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
            "earliest_validation_match_id": earliest_valid["match_id"],
            "earliest_validation_cutoff": earliest_valid["cutoff"].isoformat(),
            "latest_training_match_id": latest_train["match_id"],
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
                    "optuna",
                    "pandas",
                    "pyarrow",
                    "scikit-learn",
                    "scipy",
                    "skops",
                    "threadpoolctl",
                )
            },
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def compare_development(
    results: Sequence[MatchResult], output: Path | None = None
) -> dict:
    """Compare untuned forecasts and optionally save the fixed temporal folds."""
    if output is not None and output.exists():
        raise FileExistsError(output)
    development = development_results(results)
    frame = replay_frame(development, leak=True)
    folds = development_folds(frame)
    report, _ = _comparison(frame, folds)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as temporary:
            artifacts = Path(temporary) / output.name
            artifacts.mkdir()
            membership = artifacts / "membership.csv"
            pd.concat(_fold_membership(frame, folds)).to_csv(membership, index=False)
            report.update(
                source_sha256=SNAPSHOT_SHA256,
                data_sha256=content_sha256(results),
                development_sha256=content_sha256(development),
                validation_starts=[day.isoformat() for day in VALIDATION_STARTS],
                final_start=DEVELOPMENT_END.isoformat(),
                development_rows=len(development),
                membership_sha256=sha256_file(membership),
            )
            _write_json(artifacts / "comparison.json", report)
            artifacts.rename(output)
    return report


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


def _study_record(
    results: Sequence[MatchResult], development: Sequence[MatchResult], fold_csv: str
) -> dict:
    return {
        "source_sha256": SNAPSHOT_SHA256,
        "data_sha256": content_sha256(results),
        "development_sha256": content_sha256(development),
        "fold_membership_sha256": sha256(fold_csv.encode("utf-8")).hexdigest(),
        **_runtime_identity(),
        "config": {
            "validation_starts": [day.isoformat() for day in VALIDATION_STARTS],
            "final_start": DEVELOPMENT_END.isoformat(),
            "selection_label_cutoff": prior_day_cutoff(DEVELOPMENT_END).isoformat(),
            "features": FEATURES,
            "initial_elo": INITIAL_ELO,
            "elo_k": K,
            "search_direction": "minimize",
            "search_sampler": "TPESampler",
            "search_startup_trials": STARTUP_TRIALS,
            "search_trials": SEARCH_TRIALS,
            "search_jobs": 1,
            "c_bounds": list(C_BOUNDS),
            "c_log_scale": True,
            "seed": SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_block": "nonempty UTC calendar week starting Monday",
            "calibration_bin_edges": CALIBRATION_BINS,
            "replay_atol": REPLAY_ATOL,
            "replay_rtol": 0.0,
            "selection_rule": "Lowest mean temporal fold log loss; earliest trial on an exact tie.",
        },
    }


def _optimize(
    frame: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    experiment_id: str | None = None,
) -> tuple[dict, list[dict]]:
    import optuna

    if experiment_id is not None:
        import mlflow
    else:
        mlflow = None

    def objective(trial: Trial) -> float:
        tracking = (
            mlflow.start_run(
                experiment_id=experiment_id,
                nested=True,
                run_name=f"trial-{trial.number:02d}",
            )
            if mlflow is not None
            else nullcontext()
        )
        with tracking as run:
            c = trial.suggest_float("C", *C_BOUNDS, log=True)
            if mlflow is not None and run is not None:
                mlflow.log_params({"trial_number": trial.number, "C": c})
                trial.set_user_attr("run_id", run.info.run_id)

            fold_losses = {}
            for number, (train_idx, valid_idx) in enumerate(folds, 1):
                train, valid = frame.iloc[train_idx], frame.iloc[valid_idx]
                model = logistic_pipeline(c).fit(train[FEATURES], train["label"])
                probability = model.predict_proba(valid[FEATURES])[:, 1]
                fold_losses[f"fold_{number}_log_loss"] = log_loss(
                    valid["label"], probability, labels=[0, 1]
                )
            score = fmean(fold_losses.values())
            trial.set_user_attr("fold_losses", fold_losses)
            if mlflow is not None:
                mlflow.log_metrics({**fold_losses, "mean_log_loss": score})
            return score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=STARTUP_TRIALS),
    )
    study.optimize(objective, n_trials=SEARCH_TRIALS, n_jobs=1)
    winner = study.best_trial
    trials = [
        {
            "trial_number": trial.number,
            "C": trial.params["C"],
            **trial.user_attrs["fold_losses"],
            "mean_log_loss": trial.value,
            "status": trial.state.name,
        }
        for trial in study.trials
    ]
    decision = {
        "best_trial_number": winner.number,
        "best_params": {"model__C": winner.params["C"]},
        "best_mean_log_loss": winner.value,
    }
    if experiment_id is not None:
        decision["best_trial_run_id"] = winner.user_attrs["run_id"]
    return decision, trials


def tune_development(results: Sequence[MatchResult]) -> dict:
    """Search the fixed development folds without refitting or opening a tracking run."""
    development = development_results(results)
    frame = replay_frame(development)
    folds = development_folds(frame)
    fold_csv = pd.concat(_fold_membership(frame, folds)).to_csv(index=False)
    decision, trials = _optimize(frame, folds)
    return {
        **_study_record(results, development, fold_csv),
        **decision,
        "trials": trials,
    }


def train_candidate(results: Sequence[MatchResult], output: Path) -> dict:
    """Freeze the temporal search decision and fitted preprocessing before scoring."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    if output.exists():
        raise FileExistsError(output)
    development = development_results(results)
    frame = replay_frame(development, leak=True)
    folds = development_folds(frame)
    fold_membership = _fold_membership(frame, folds)
    fold_csv = pd.concat(fold_membership).to_csv(index=False)
    study_record = _study_record(results, development, fold_csv)
    comparison, random_membership = _comparison(frame, folds)
    metrics = comparison["mean_log_loss"]
    refit_membership = frame[MEMBERSHIP].assign(split="final_refit", partition="train")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / output.name
        bundle.mkdir()
        with mlflow.start_run(
            experiment_id=_tracking_experiment(output.parent), run_name="selection"
        ) as run:
            mlflow.log_dict(study_record, "study.json")
            mlflow.log_params(
                {
                    "data_sha256": study_record["data_sha256"],
                    "fold_membership_sha256": study_record["fold_membership_sha256"],
                    "seed": SEED,
                    "search_direction": study_record["config"]["search_direction"],
                    "search_trials": SEARCH_TRIALS,
                }
            )
            decision, trials = _optimize(
                frame, folds, experiment_id=run.info.experiment_id
            )
            metrics["logistic_tuned"] = decision["best_mean_log_loss"]
            model = logistic_pipeline(decision["best_params"]["model__C"]).fit(
                frame[FEATURES], frame["label"]
            )
            joblib.dump(model, bundle / "model.joblib")
            pd.concat([refit_membership, *fold_membership, random_membership]).to_csv(
                bundle / "membership.csv", index=False
            )
            pd.DataFrame(trials).to_csv(bundle / "search.csv", index=False)

            example = frame[FEATURES].head(2)
            model_info = mlflow.sklearn.log_model(
                model,
                name="historical-logistic",
                signature=infer_signature(example, model.predict_proba(example)),
                pyfunc_predict_fn="predict_proba",
                serialization_format="skops",
                # scikit-learn 1.6 stores this class in the fitted ColumnTransformer.
                skops_trusted_types=[
                    "sklearn.compose._column_transformer._RemainderColsList"
                ],
                pip_requirements=[
                    f"{name}=={version(name)}"
                    for name in ("scikit-learn", "numpy", "pandas", "skops")
                ],
                params=decision["best_params"],
                tags={"best_trial_run_id": decision["best_trial_run_id"]},
            )
            record = {
                "run_id": run.info.run_id,
                **study_record,
                **decision,
                "logged_model_uri": model_info.model_uri,
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
                    **decision["best_params"],
                    "best_trial_number": record["best_trial_number"],
                    "model_sha256": record["artifacts_sha256"]["model.joblib"],
                }
            )
            mlflow.set_tag("best_trial_run_id", record["best_trial_run_id"])
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
    for name in ("compare", "tune", "train", "evaluate"):
        subparser = commands.add_parser(name)
        subparser.add_argument("results", type=Path)
        if name == "compare":
            subparser.add_argument(
                "--output",
                type=Path,
                help="Save comparison and fold membership in a new directory.",
            )
        elif name != "tune":
            subparser.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)
    results = read_results(args.results)
    if args.command == "compare":
        report = compare_development(results, args.output)
    elif args.command == "tune":
        report = tune_development(results)
    elif args.command == "train":
        report = train_candidate(results, args.bundle)
    else:
        report = evaluate_candidate(results, args.bundle)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
