"""Executable evidence for the replay-clock lesson."""

from contextlib import closing
import json
from math import log1p
from pathlib import Path
import sqlite3

import pytest

from m8_replay import import_capture, report

CAPTURE = Path(__file__).with_name("fixtures") / "m8-incident.jsonl"
BEFORE = "2030-06-11T23:59:59Z"
AFTER = "2030-06-12T00:00:00Z"


def test_replay_clock_separates_requests_forecasts_and_mature_labels():
    with closing(sqlite3.connect(":memory:")) as db:
        import_capture(db, CAPTURE)
        baseline = report(db, "2030-06-10T12:00:01Z")
        pending = report(db, BEFORE)
        mature = report(db, "2030-06-11T20:00:00-04:00")
        import_capture(db, CAPTURE)
        assert report(db, AFTER) == mature
        assert report(db, BEFORE) == pending

    assert baseline["request_count"] == 1
    assert baseline["quality"][0]["forecast_count"] == 1
    assert mature["request_count"] == pending["request_count"] == 4
    assert mature["error_count"] == 1
    assert mature["surface_mix"] == [
        {"replay_day": "2030-06-10", "surface": "Clay", "replies": 1},
        {"replay_day": "2030-06-11", "surface": "Hard", "replies": 2},
    ]
    _, slow, retry, error = mature["requests"]
    assert slow["forecast_id"] == retry["forecast_id"]
    assert slow["request_id"] != retry["request_id"]
    assert slow["duration_ms"] >= slow["lookup_ms"] >= 250
    assert retry["lookup_ms"] < slow["lookup_ms"]
    assert error["status_code"] == 422
    assert error["lookup_ms"] is error["forecast_id"] is None

    [unscored] = pending["quality"]
    [scored] = mature["quality"]
    assert scored["forecast_count"] == unscored["forecast_count"] == 2
    assert unscored["labeled_count"] == unscored["label_coverage"] == 0
    assert unscored["log_loss"] is unscored["calibration"] is None
    assert scored["labeled_count"] == 1
    assert scored["label_coverage"] == 0.5
    [hard] = [
        forecast for forecast in scored["forecasts"] if forecast["label"] is not None
    ]
    assert (hard["fixture_id"], hard["p1_id"], hard["label"]) == ("424816", "1985", 0)
    assert hard["p1_win_probability"] == pytest.approx(0.527281, abs=1e-6)
    assert scored["log_loss"] == pytest.approx(-log1p(-hard["p1_win_probability"]))
    [occupied] = [row for row in scored["calibration"] if row["matches"]]
    assert occupied["matches"] == 1
    assert occupied["observed_win_rate"] == 0


def test_conflicting_import_rolls_back_new_events(tmp_path):
    records = [json.loads(line) for line in CAPTURE.read_text().splitlines()]
    request = next(row for row in records if row["event"] == "http_request_completed")
    outcome = next(row for row in records if row["event"] == "forecast_outcome")
    conflicting = tmp_path / "conflicting.jsonl"
    conflicting.write_text(
        json.dumps({**request, "request_id": "new-attempt"})
        + "\n"
        + json.dumps({**outcome, "label": 1 - outcome["label"]})
        + "\n"
    )
    with closing(sqlite3.connect(":memory:")) as db:
        import_capture(db, CAPTURE)
        original = report(db, AFTER)
        with pytest.raises(ValueError, match="Conflicting outcome"):
            import_capture(db, conflicting)
        assert report(db, AFTER) == original


def test_reverse_order_reply_can_arrive_first(tmp_path):
    records = [json.loads(line) for line in CAPTURE.read_text().splitlines()]
    forward, reverse = [
        row for row in records if row.get("forecast", {}).get("fixture_id") == "424816"
    ]
    for row in records:
        if row.get("request_id") == forward["request_id"]:
            row["observed_at"] = "2030-06-12T00:00:01Z"
    reordered = tmp_path / "reverse-first.jsonl"
    reordered.write_text("".join(json.dumps(row) + "\n" for row in records))

    with closing(sqlite3.connect(":memory:")) as db:
        import_capture(db, reordered)
        before_forward = report(db, AFTER)

    assert before_forward["request_count"] == 3
    [quality] = before_forward["quality"]
    assert quality["forecast_count"] == 2
    [scored] = [row for row in quality["forecasts"] if row["label"] is not None]
    assert scored["forecast_id"] == reverse["forecast_id"]
    assert scored["p1_id"] == forward["forecast"]["p1_id"]
    assert scored["p1_win_probability"] == pytest.approx(
        1 - reverse["forecast"]["p1_win_probability"]
    )
