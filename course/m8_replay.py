"""Import a small forecast capture into SQLite and report as of a replay time."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
import sqlite3

import numpy as np

from acebet.historical_monitoring import read_forecasts
from acebet.train.historical_evaluation import _calibration, _losses

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    kind TEXT NOT NULL,
    event_key TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (kind, event_key)
)
"""

REQUESTS = """
SELECT request.event_key AS request_id, request.observed_at,
       request.payload ->> 'status_code' AS status_code,
       request.payload ->> 'duration_ms' AS duration_ms,
       lookup.payload ->> 'duration_ms' AS lookup_ms,
       forecast.payload ->> 'forecast_id' AS forecast_id,
       forecast.payload ->> '$.forecast.surface' AS surface
FROM events AS request
LEFT JOIN events AS lookup
    ON lookup.kind = 'lookup' AND lookup.event_key = request.event_key
    AND lookup.observed_at <= :as_of
LEFT JOIN events AS forecast
    ON forecast.kind = 'forecast' AND forecast.event_key = request.event_key
    AND forecast.observed_at <= :as_of
WHERE request.kind = 'request' AND request.observed_at <= :as_of
"""

SURFACE_MIX = f"""
SELECT substr(observed_at, 1, 10) AS replay_day, surface, count(*) AS replies
FROM ({REQUESTS})
WHERE status_code = 200 AND surface IS NOT NULL
GROUP BY replay_day, surface
ORDER BY replay_day, surface
"""

QUALITY_ROWS = """
WITH visible_forecasts AS (
    SELECT payload ->> 'forecast_id' AS forecast_id,
           payload -> '$.forecast' AS forecast,
           row_number() OVER (
               PARTITION BY payload ->> 'forecast_id'
               ORDER BY observed_at, event_key
           ) AS attempt
    FROM events
    WHERE kind = 'forecast' AND observed_at <= :as_of
        AND payload ->> '$.forecast.cutoff' <= :as_of
)
SELECT f.forecast_id, f.forecast, outcome.payload ->> 'label' AS label
FROM visible_forecasts AS f
LEFT JOIN events AS outcome
    ON outcome.kind = 'outcome'
    AND outcome.observed_at <= :as_of
    AND outcome.event_key = (f.forecast ->> 'fixture_id')
    AND (outcome.payload ->> 'p1_id') = (f.forecast ->> 'p1_id')
    AND (outcome.payload ->> 'p2_id') = (f.forecast ->> 'p2_id')
    AND (outcome.payload ->> 'surface') = (f.forecast ->> 'surface')
    AND (outcome.payload ->> 'cutoff') = (f.forecast ->> 'cutoff')
WHERE f.attempt = 1
ORDER BY f.forecast_id
"""

REQUEST_EVENTS = {"http_request_completed", "http_request_failed"}
LOG_EVENTS = REQUEST_EVENTS | {"forecast_served", "feature_lookup_completed"}


def utc_key(timestamp: str) -> str:
    """Make timestamps sortable as text, including offsets and fractional seconds."""
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("Use a timestamp with an explicit UTC offset.")
    return parsed.astimezone(timezone.utc).isoformat(timespec="microseconds")


def import_capture(db: sqlite3.Connection, capture: Path) -> None:
    """Insert new events atomically and verify repeated keys agree."""
    with db, capture.open(encoding="utf-8") as lines:
        db.execute("BEGIN IMMEDIATE")
        db.execute(SCHEMA)
        for line in lines:
            entry = json.loads(line)
            event = entry["event"]
            at = utc_key(entry["observed_at"])
            if event == "forecast_outcome":
                kind, key = "outcome", entry["fixture_id"]
                entry["cutoff"] = utc_key(entry["cutoff"])
                if (
                    type(entry["label"]) is not int
                    or entry["label"] not in (0, 1)
                    or entry["p1_id"] >= entry["p2_id"]
                    or at <= entry["cutoff"]
                ):
                    raise ValueError(f"Invalid outcome for {key}.")
            elif event == "forecast_served":
                kind, key = "forecast", entry["request_id"]
                # ponytail: scan this small capture; index the forecast ID for larger imports.
                recorded = db.execute(
                    """SELECT payload FROM events
                       WHERE kind = 'forecast' AND (payload ->> 'forecast_id') = ?
                       LIMIT 1""",
                    (entry["forecast_id"],),
                ).fetchone()
                replies = [line]
                if recorded:
                    replies.append(recorded[0])
                [response] = read_forecasts(replies).values()
                entry["forecast"] = response.model_dump(mode="json")
                entry["forecast"]["cutoff"] = utc_key(response.cutoff.isoformat())
            elif event in REQUEST_EVENTS or event == "feature_lookup_completed":
                kind = "request" if event in REQUEST_EVENTS else "lookup"
                key = entry["request_id"]
                duration = entry["duration_ms"]
                if not isfinite(duration) or duration < 0:
                    raise ValueError(f"Invalid duration for {key}.")
                if kind == "request" and not 100 <= entry["status_code"] <= 599:
                    raise ValueError(f"Invalid HTTP status for {key}.")
            else:
                raise ValueError(f"Unexpected replay event {event}.")

            entry["observed_at"] = at
            payload = json.dumps(entry, sort_keys=True, allow_nan=False)
            cursor = db.execute(
                """INSERT INTO events VALUES (?, ?, ?, ?)
                   ON CONFLICT (kind, event_key) DO NOTHING""",
                (kind, key, at, payload),
            )
            if cursor.rowcount == 0:
                previous_payload = db.execute(
                    "SELECT payload FROM events WHERE kind = ? AND event_key = ?",
                    (kind, key),
                ).fetchone()[0]
                if payload != previous_payload:
                    raise ValueError(f"Conflicting {kind} event for {key}.")


def report(db: sqlite3.Connection, as_of: str) -> dict:
    """Count visible requests and score eligible unique forecasts using M4c's metrics."""
    params = {"as_of": utc_key(as_of)}
    db.row_factory = sqlite3.Row
    requests = [
        dict(row)
        for row in db.execute(
            REQUESTS + " ORDER BY request.observed_at, request_id", params
        )
    ]
    surface_mix = [dict(row) for row in db.execute(SURFACE_MIX, params)]
    groups = defaultdict(list)
    for row in db.execute(QUALITY_ROWS, params):
        forecast = json.loads(row["forecast"])
        groups[forecast["model_sha256"], forecast["history_sha256"]].append(
            {
                "forecast_id": row["forecast_id"],
                "fixture_id": forecast["fixture_id"],
                "cutoff": forecast["cutoff"],
                "p1_id": forecast["p1_id"],
                "p1_win_probability": forecast["p1_win_probability"],
                "feature_state_sha256": forecast["feature_state_sha256"],
                "label": row["label"],
            }
        )
    quality = []
    for (model, history), forecasts in sorted(groups.items()):
        scored = [row for row in forecasts if row["label"] is not None]
        labels = np.array([row["label"] for row in scored], dtype=int)
        probabilities = np.array([row["p1_win_probability"] for row in scored])
        quality.append(
            {
                "model_sha256": model,
                "history_sha256": history,
                "forecast_count": len(forecasts),
                "labeled_count": len(scored),
                "label_coverage": len(scored) / len(forecasts),
                "log_loss": float(_losses(labels, probabilities).mean())
                if scored
                else None,
                "calibration": _calibration(labels, probabilities) if scored else None,
                "forecasts": forecasts,
            }
        )
    return {
        "as_of": params["as_of"],
        "request_count": len(requests),
        "error_count": sum(row["status_code"] >= 400 for row in requests),
        "requests": requests,
        "surface_mix": surface_mix,
        "quality": quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("as_of")
    args = parser.parse_args()
    as_of = utc_key(args.as_of)
    with closing(sqlite3.connect(args.database)) as db:
        import_capture(db, args.capture)
        with db:
            db.execute("BEGIN")
            print(json.dumps(report(db, as_of), indent=2))


if __name__ == "__main__":
    main()
