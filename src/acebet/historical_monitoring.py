"""Score recorded HTTP forecasts as labels become available, or check a candidate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
import json
from math import isclose
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import AwareDatetime, TypeAdapter

from acebet.dataprep.historical_replay import prior_day_cutoff
from acebet.dataprep.historical_results import (
    MatchResult,
    content_sha256,
    read_results,
)
from acebet.historical_forecast import (
    ForecastRequest,
    ForecastResponse,
    forecast_id,
    load_forecaster,
)
from acebet.model_artifacts import read_json, sha256_file, write_json
from acebet.train.historical_evaluation import (
    _calibration,
    _losses,
    paired_week_interval,
)

MODEL_SCORES = ("prevalence", "elo", "logistic_tuned")


def read_forecasts(lines: Iterable[str]) -> dict[str, ForecastResponse]:
    """Read the JSON request log and deduplicate repeated fixture forecasts."""
    forecasts: dict[str, ForecastResponse] = {}
    for line in lines:
        entry = json.loads(line)
        if entry.get("event") != "forecast_served":
            continue
        forecast = ForecastResponse.model_validate(entry["forecast"])
        key = forecast_id(forecast)
        if entry["forecast_id"] != key:
            raise ValueError(f"Forecast ID differs from its recorded identity: {key}")
        if forecast.p1_id > forecast.p2_id:
            forecast = forecast.model_copy(
                update={
                    "p1_id": forecast.p2_id,
                    "p2_id": forecast.p1_id,
                    "elo_gap": -forecast.elo_gap,
                    "p1_win_probability": 1 - forecast.p1_win_probability,
                }
            )
        previous = forecasts.setdefault(key, forecast)
        if previous.elo_gap != forecast.elo_gap or not isclose(
            previous.p1_win_probability,
            forecast.p1_win_probability,
            rel_tol=0,
            abs_tol=1e-15,
        ):
            raise ValueError(f"Conflicting forecasts for ID {key}")
    return forecasts


def quality_report(
    forecasts: Iterable[ForecastResponse],
    results: Sequence[MatchResult],
    as_of: datetime,
) -> dict:
    """Score only results available at the simulated observation time."""
    if as_of.utcoffset() is None:
        raise ValueError("The report time must have a UTC offset.")
    available = {
        match.match_id: match for match in results if match.available_at <= as_of
    }
    groups = defaultdict(list)
    for forecast in forecasts:
        if forecast.cutoff > as_of:
            continue
        label = None
        match = available.get(forecast.fixture_id)
        if match is not None:
            players = (forecast.p1_id, forecast.p2_id)
            expected = (match.p1_id, match.p2_id)
            if (
                tuple(sorted(players)) != expected
                or forecast.surface != match.surface
                or forecast.cutoff != prior_day_cutoff(match.event_date)
            ):
                raise ValueError(f"Incompatible result for {forecast.fixture_id}")
            label = match.label if players == expected else 1 - match.label
        groups[
            (
                forecast.cutoff.strftime("%Y-%m"),
                forecast.model_sha256,
                forecast.history_sha256,
            )
        ].append((forecast.p1_win_probability, label))

    periods = []
    for (period, model, history), rows in sorted(groups.items()):
        scored = [
            (probability, label) for probability, label in rows if label is not None
        ]
        probabilities = np.array([probability for probability, _ in scored])
        labels = np.array([label for _, label in scored], dtype=int)
        periods.append(
            {
                "cutoff_month_utc": period,
                "model_sha256": model,
                "history_sha256": history,
                "forecasts": len(rows),
                "labeled": len(scored),
                "label_coverage": len(scored) / len(rows),
                "log_loss": float(_losses(labels, probabilities).mean())
                if scored
                else None,
                "calibration": _calibration(labels, probabilities) if scored else None,
            }
        )
    return {
        "as_of": as_of.isoformat(),
        "results_sha256": content_sha256(results),
        "periods": periods,
    }


def promotion_check(final: Path, candidate: str, incumbent: str) -> dict:
    """Require lower paired log loss on the same fixtures before approval."""
    if candidate == incumbent:
        raise ValueError("Choose two different scores from the recorded comparison.")
    evaluation = read_json(final / "evaluation.json")
    predictions = final / "forecasts.csv"
    if sha256_file(predictions) != evaluation["forecasts_sha256"]:
        raise ValueError("Forecasts differ from the recorded final evaluation.")
    frame = pd.read_csv(predictions)
    if len(frame) != evaluation["matches"] or not frame["match_id"].is_unique:
        raise ValueError("Promotion requires one score per recorded match.")
    label = frame["label"].to_numpy()
    probability = frame[[candidate, incumbent]].to_numpy(dtype=float)
    if (
        not np.isin(label, [0, 1]).all()
        or not ((probability >= 0) & (probability <= 1)).all()
    ):
        raise ValueError(
            "Promotion scores must have binary labels and valid probabilities."
        )
    delta = _losses(label, probability[:, 0]) - _losses(label, probability[:, 1])
    evidence = paired_week_interval(delta, pd.to_datetime(frame["event_date"]).dt.date)
    return {
        "candidate": candidate,
        "incumbent": incumbent,
        "matches": len(frame),
        "forecasts_sha256": evaluation["forecasts_sha256"],
        **evidence,
        "accepted": evidence["interval_95"][1] < 0,
    }


def review_candidate(results: Path, bundle: Path) -> dict:
    """Bind the gate decision and fixture replay to one frozen deployment."""
    selection_path = bundle / "selection.json"
    selection_hash = sha256_file(selection_path)
    selection = read_json(selection_path)
    evaluation_path = bundle / "final/evaluation.json"
    evaluation = read_json(evaluation_path)
    forecaster = load_forecaster(results, bundle)
    if (
        evaluation["selection_sha256"],
        evaluation["selection_run_id"],
        evaluation["model_sha256"],
    ) != (selection_hash, selection["run_id"], forecaster.model_sha256):
        raise ValueError("Final evaluation does not belong to the frozen selection.")
    fixture = ForecastRequest.model_validate(
        {key: selection["fixture"][key] for key in ForecastRequest.model_fields}
    )
    return {
        "selection_sha256": selection_hash,
        "evaluation_sha256": sha256_file(evaluation_path),
        "fixture": forecaster.predict(fixture).model_dump(mode="json"),
        "gate": promotion_check(bundle / "final", "logistic_tuned", "elo"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    quality = subcommands.add_parser("quality")
    quality.add_argument("log", type=Path)
    quality.add_argument("results", type=Path)
    quality.add_argument("as_of", type=TypeAdapter(AwareDatetime).validate_python)
    gate = subcommands.add_parser("gate")
    gate.add_argument("final", type=Path)
    gate.add_argument("--candidate", choices=MODEL_SCORES, required=True)
    gate.add_argument("--incumbent", choices=MODEL_SCORES, required=True)
    for command in ("review", "promote"):
        candidate = subcommands.add_parser(command)
        candidate.add_argument("results", type=Path)
        candidate.add_argument("bundle", type=Path)
    args = parser.parse_args()
    rejected = False
    if args.command == "quality":
        with args.log.open(encoding="utf-8") as stream:
            forecasts = read_forecasts(stream)
        report = quality_report(
            forecasts.values(), read_results(args.results), args.as_of
        )
    elif args.command == "gate":
        report = promotion_check(args.final, args.candidate, args.incumbent)
        rejected = not report["accepted"]
    else:
        report = review_candidate(args.results, args.bundle)
        if args.command == "promote":
            rejected = not report["gate"]["accepted"]
            if not rejected:
                report["approved_at"] = datetime.now(timezone.utc).isoformat()
                write_json(args.bundle / "promotion.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    if rejected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
