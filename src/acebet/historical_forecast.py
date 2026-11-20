"""Replay requests against the frozen M4 model and its recorded feature state."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
from math import isclose
from pathlib import Path
import sys
from typing import Annotated, Literal, cast

import joblib
import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator
from sklearn.pipeline import Pipeline

from acebet.dataprep.historical_replay import (
    DEVELOPMENT_END,
    F01_SURFACE,
    INITIAL_ELO,
    K,
    prior_day_cutoff,
    ratings_as_of,
    worked_fixture,
)
from acebet.dataprep.historical_results import (
    SNAPSHOT_SHA256,
    MatchResult,
    content_sha256,
    read_results,
)
from acebet.forecast import F01_CUTOFF, F01_ID, F01_PLAYERS
from acebet.model_artifacts import read_json, sha256_file
from acebet.train.historical_evaluation import FEATURES, REPLAY_ATOL

ReplayID = Annotated[str, Field(strict=True, pattern=r"^[A-Za-z0-9-]{1,64}$")]


class InvalidReplay(ValueError):
    """The request does not match a fixture the frozen model can replay."""


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fixture_id: ReplayID
    p1_id: ReplayID
    p2_id: ReplayID
    surface: Literal["Clay", "Grass", "Hard"]
    cutoff: AwareDatetime

    @field_validator("cutoff", mode="before")
    @classmethod
    def require_timestamp(cls, value: object) -> object:
        if not isinstance(value, (datetime, str)) or (
            isinstance(value, str) and "T" not in value
        ):
            raise ValueError("Provide a timestamp with a UTC offset.")
        return value

    @field_validator("cutoff")
    @classmethod
    def normalize_cutoff(cls, value: datetime) -> datetime:
        try:
            return value.astimezone(timezone.utc)
        except OverflowError as exc:
            raise ValueError("Cutoff is outside the representable UTC range.") from exc


class ForecastResponse(ForecastRequest):
    model_config = ConfigDict(protected_namespaces=())

    p1_win_probability: Annotated[float, Field(ge=0, le=1)]
    elo_gap: float
    model_sha256: str
    history_sha256: str
    feature_state_sha256: str


def forecast_id(response: ForecastResponse) -> str:
    """Identify one model and feature state for the same fixture and cutoff."""
    identity = response.model_dump(
        mode="json", exclude={"elo_gap", "p1_win_probability"}
    )
    identity["p1_id"], identity["p2_id"] = sorted((response.p1_id, response.p2_id))
    return sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class FrozenForecast:
    model: Pipeline
    results: tuple[MatchResult, ...]
    matches: dict[str, MatchResult]
    model_sha256: str
    history_sha256: str

    def predict(
        self,
        request: ForecastRequest,
        *,
        span: Callable[[str], AbstractContextManager[object]] = nullcontext,
    ) -> ForecastResponse:
        with span("forecast.feature_lookup"):
            cutoff = request.cutoff
            players = tuple(sorted((request.p1_id, request.p2_id)))
            if request.fixture_id == F01_ID:
                expected = (F01_PLAYERS, F01_SURFACE, F01_CUTOFF)
                observed = (
                    players,
                    request.surface,
                    cutoff.isoformat().replace("+00:00", "Z"),
                )
                if observed != expected:
                    raise InvalidReplay(
                        "F01 requires its recorded players, surface, and cutoff."
                    )
                fixture = worked_fixture()
                gap = cast(float, fixture["p1_elo"]) - cast(float, fixture["p2_elo"])
                feature_state = sha256(
                    json.dumps(fixture, sort_keys=True).encode()
                ).hexdigest()
            else:
                match = self.matches.get(request.fixture_id)
                if match is None or match.event_date < DEVELOPMENT_END:
                    raise InvalidReplay(
                        "Choose F01 or a fixture from the recorded final period."
                    )
                if (players, request.surface, cutoff) != (
                    (match.p1_id, match.p2_id),
                    match.surface,
                    prior_day_cutoff(match.event_date),
                ):
                    raise InvalidReplay(
                        "Use the fixture's recorded players, surface, and cutoff."
                    )
                # ponytail: replay this small extract in memory, add checkpoints when measured traffic needs them.
                ratings = ratings_as_of(self.results, cutoff)
                gap = ratings.get(match.p1_id, INITIAL_ELO) - ratings.get(
                    match.p2_id, INITIAL_ELO
                )
                feature_state = self.history_sha256

            inputs = pd.DataFrame({"elo_gap": [gap], "surface": [request.surface]})
        with span("forecast.inference"):
            probability = float(self.model.predict_proba(inputs)[0, 1])
        if request.p1_id != players[0]:
            gap, probability = -gap, 1.0 - probability
        return ForecastResponse(
            fixture_id=request.fixture_id,
            p1_id=request.p1_id,
            p2_id=request.p2_id,
            surface=request.surface,
            cutoff=cutoff,
            elo_gap=gap,
            p1_win_probability=probability,
            model_sha256=self.model_sha256,
            history_sha256=self.history_sha256,
            feature_state_sha256=feature_state,
        )


def load_forecaster(results_path: Path, bundle: Path) -> FrozenForecast:
    """Validate M4's history, metadata, and model bytes before serving."""
    selection = read_json(bundle / "selection.json")
    results = tuple(read_results(results_path))
    history_hash = content_sha256(results)
    if (selection["source_sha256"], selection["data_sha256"]) != (
        SNAPSHOT_SHA256,
        history_hash,
    ):
        raise ValueError("Canonical results differ from the frozen selection.")

    config = selection["config"]
    for name, expected in {
        "features": FEATURES,
        "initial_elo": INITIAL_ELO,
        "elo_k": K,
        "final_start": DEVELOPMENT_END.isoformat(),
        "selection_label_cutoff": prior_day_cutoff(DEVELOPMENT_END).isoformat(),
    }.items():
        if config[name] != expected:
            raise ValueError(f"Forecast feature contract differs at {name}.")
    package = Path(__file__).resolve().parent
    for name in (
        "dataprep/historical_results.py",
        "dataprep/historical_replay.py",
        "features.py",
        "forecast.py",
    ):
        if selection["sources_sha256"][name] != sha256_file(package / name):
            raise ValueError(f"Forecast feature source differs at {name}.")
    dependencies = selection["dependencies"]
    if dependencies["python"].split(".")[:2] != list(map(str, sys.version_info[:2])):
        raise ValueError(
            "Python major and minor version must match the frozen selection."
        )
    for name in (
        "joblib",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "threadpoolctl",
    ):
        if dependencies[name] != version(name):
            raise ValueError(f"Serving dependency differs at {name}.")

    model_bytes = (bundle / "model.joblib").read_bytes()
    model_hash = sha256(model_bytes).hexdigest()
    if model_hash != selection["artifacts_sha256"]["model.joblib"]:
        raise ValueError("model.joblib differs from the frozen selection.")
    # Only load locally produced bundles from a trusted training run.
    model = joblib.load(BytesIO(model_bytes))
    if (
        not isinstance(model, Pipeline)
        or list(model.feature_names_in_) != FEATURES
        or list(model.classes_) != [0, 1]
    ):
        raise ValueError(
            "The fitted pipeline must accept the recorded features and classes."
        )

    forecaster = FrozenForecast(
        model, results, {row.match_id: row for row in results}, model_hash, history_hash
    )
    fixture = selection["fixture"]
    if (
        any(fixture[key] != value for key, value in worked_fixture().items())
        or fixture["surface"] != F01_SURFACE
    ):
        raise ValueError("F01 feature trace differs from the frozen selection.")
    replay = forecaster.predict(
        ForecastRequest.model_validate(
            {key: fixture[key] for key in ForecastRequest.model_fields}
        )
    )
    if not isclose(
        replay.p1_win_probability,
        fixture["logistic_p1_win_probability"],
        rel_tol=0,
        abs_tol=REPLAY_ATOL,
    ):
        raise ValueError("F01 probability differs from the frozen selection.")
    return forecaster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    request = ForecastRequest.model_validate_json(sys.stdin.read())
    print(load_forecaster(args.results, args.bundle).predict(request).model_dump_json())


if __name__ == "__main__":
    main()
