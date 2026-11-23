"""Record a reproducible incident from an M7 candidate in a disposable process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
from unittest.mock import patch

from fastapi.testclient import TestClient

from acebet import historical_forecast
from acebet.app.historical import create_app
from acebet.app.observability import get_request_logger
from acebet.dataprep.historical_replay import F01_SURFACE, prior_day_cutoff
from acebet.dataprep.historical_results import read_results
from acebet.forecast import F01_CUTOFF, F01_ID, F01_PLAYERS

from m8_replay import LOG_EVENTS


def capture(candidate: Path, output: Path) -> None:
    """Record real replies and durations, then attach scheduled replay times."""
    results_path = candidate / "matches.parquet"
    match = next(row for row in read_results(results_path) if row.match_id == "424816")
    clay = dict(
        fixture_id=F01_ID,
        p1_id=F01_PLAYERS[0],
        p2_id=F01_PLAYERS[1],
        surface=F01_SURFACE,
        cutoff=F01_CUTOFF,
    )
    hard = dict(
        fixture_id=match.match_id,
        p1_id=match.p1_id,
        p2_id=match.p2_id,
        surface=match.surface,
        cutoff=prior_day_cutoff(match.event_date).isoformat(),
    )
    attempts = (
        ("2030-06-10T12:00:01Z", clay, 200, 0.0),
        ("2030-06-11T12:00:01Z", hard, 200, 0.25),
        (
            "2030-06-11T12:00:02Z",
            {**hard, "p1_id": match.p2_id, "p2_id": match.p1_id},
            200,
            0.0,
        ),
        ("2030-06-11T12:00:03Z", {**hard, "surface": "Ice"}, 422, 0.0),
    )
    times = {}
    rating_lookup = historical_forecast.ratings_as_of
    delay_seconds = 0.0

    def timed_lookup(results, cutoff):
        start = perf_counter()
        if delay_seconds:
            sleep(delay_seconds)
        ratings = rating_lookup(results, cutoff)
        get_request_logger().info(
            "feature_lookup_completed",
            duration_ms=round((perf_counter() - start) * 1000, 2),
        )
        return ratings

    with TemporaryDirectory() as scratch:
        log = Path(scratch) / "requests.jsonl"
        settings = {
            "ACEBET_FORECAST_RESULTS": str(results_path),
            "ACEBET_FORECAST_BUNDLE": str(candidate / "bundle"),
            "ACEBET_LOG_FILE": str(log),
            "ACEBET_LOG_FORMAT": "json",
            "ACEBET_LOG_LEVEL": "INFO",
            "ACEBET_METRICS_ENABLED": "false",
            "ACEBET_TRACING_ENABLED": "false",
        }
        with (
            patch.dict("os.environ", settings),
            patch.object(historical_forecast, "ratings_as_of", timed_lookup),
            TestClient(create_app()) as client,
        ):
            for observed_at, request, status, delay_seconds in attempts:
                response = client.post("/forecast", json=request)
                if response.status_code != status:
                    raise RuntimeError(
                        f"Expected HTTP {status}, got {response.status_code}: "
                        f"{response.text}\n{log.read_text(encoding='utf-8')}"
                    )
                times[response.headers["X-Request-ID"]] = observed_at

        events = []
        for line in log.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            request_id = entry.get("request_id")
            if entry["event"] in LOG_EVENTS and request_id in times:
                entry["observed_at"] = times[request_id]
                events.append(entry)

    events.append(
        {
            "event": "forecast_outcome",
            "observed_at": "2030-06-12T00:00:00Z",
            "archive_available_at": match.available_at.isoformat(),
            "label": match.label,
            **hard,
        }
    )
    with output.open("x", encoding="utf-8") as file:
        for event in events:
            file.write(json.dumps(event, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    capture(args.candidate, args.output)


if __name__ == "__main__":
    main()
