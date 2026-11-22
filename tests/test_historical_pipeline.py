"""M3 evidence for source reconstruction and point-in-time replay."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from acebet.dataprep import historical_results
from acebet.dataprep.historical_replay import (
    WinnerCorrection,
    development_scores,
    ratings_as_of,
    replay_matches,
    worked_fixture,
)
from acebet.dataprep.historical_results import (
    MatchResult,
    build_results,
    content_sha256,
    normalize_results,
    read_results,
)

SNAPSHOT = Path(__file__).resolve().parents[1] / "data/raw/valuebetennis-2021-atp.csv"


def test_rebuild_preserves_source_and_reproduces_canonical_content(tmp_path):
    output = tmp_path / "matches.parquet"
    rebuilt = tmp_path / "rebuilt.parquet"
    raw = SNAPSHOT.read_bytes()
    first = build_results(SNAPSHOT, output)
    matches = read_results(output)
    assert (
        build_results(SNAPSHOT, rebuilt)
        == first
        == {
            "source_sha256": "ca16daa1c89b085680152dc41924cad7e2791775c23d68a083c2c0237cdf0598",
            "content_sha256": "2336bd8a6d63d2f88919b3956423183c5f33c1d09ad913acdc9255c82c83b71d",
            "rows": 1856,
            "input_rows": 2091,
            "duplicate_rows": 0,
            "excluded_rows": 235,
        }
    )
    assert read_results(rebuilt) == matches
    first_match = next(row for row in matches if row.match_id == "369751")
    assert (first_match.p1_id, first_match.p2_id, first_match.label) == (
        "12066",
        "9837",
        0,
    )
    assert first_match.event_date == date(2021, 1, 7)
    assert first_match.available_at == datetime(2021, 1, 9, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="preserve the raw snapshot"):
        build_results(SNAPSHOT, SNAPSHOT)
    modified_snapshot = tmp_path / "modified.csv"
    modified_snapshot.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="raw snapshot SHA-256"):
        build_results(modified_snapshot, output)
    assert read_results(output) == matches
    assert SNAPSHOT.read_bytes() == raw


def test_raw_repeats_conflicts_and_incomplete_scores():
    header, row, *_ = SNAPSHOT.read_bytes().splitlines(keepends=True)
    original = normalize_results(header + row)
    assert normalize_results(header + row + row) == original
    conflict = row.replace(b'"6-0 6-4"', b'"6-4 6-4"')
    assert conflict != row
    with pytest.raises(ValueError, match="Conflicting duplicate for match 369751"):
        normalize_results(header + row + conflict)
    for score in (b"", b"6-4 3-1", b"6-4 ret.", b"4-6 4-6", b"6-4 6-4 4-6"):
        incomplete = row.replace(b"369751;", b"369799;", 1).replace(
            b'"6-0 6-4"', b'"' + score + b'"'
        )
        assert normalize_results(header + row + incomplete) == original
    with pytest.raises(ValueError, match="Invalid match, player, or winner identity"):
        normalize_results(header + row.replace(b";12066;", b";9837;"))
    with pytest.raises(ValueError, match="Unexpected Valuebetennis CSV columns"):
        normalize_results(header.replace(b"joueur1_id", b"bad_header") + row)


def test_preparation_report_counts_repeats_separately_from_exclusions(
    tmp_path, monkeypatch
):
    header, row, *_ = SNAPSHOT.read_bytes().splitlines(keepends=True)
    incomplete = row.replace(b"369751;", b"369799;", 1).replace(
        b'"6-0 6-4"', b'"6-0 3-1"'
    )
    raw = header + row + row + incomplete + incomplete
    snapshot = tmp_path / "snapshot.csv"
    snapshot.write_bytes(raw)
    monkeypatch.setattr(historical_results, "SNAPSHOT_SHA256", sha256(raw).hexdigest())

    report = build_results(snapshot, tmp_path / "matches.parquet")

    assert report["input_rows"] == 4
    assert report["rows"] == 1
    assert report["duplicate_rows"] == 2
    assert report["excluded_rows"] == 1


def test_changed_input_or_availability_makes_old_output_stale(tmp_path):
    output = tmp_path / "matches.parquet"
    built = build_results(SNAPSHOT, output)
    matches = read_results(output)
    first = matches[0]
    changed = replace(first, available_at=first.available_at + timedelta(days=1))
    assert content_sha256([changed, *matches[1:]]) != built["content_sha256"]
    assert content_sha256(reversed(matches)) == built["content_sha256"]

    table = pq.ParquetFile(output).read()
    labels = pa.array([1 - first.label, *(row.label for row in matches[1:])], pa.int8())
    index = table.schema.get_field_index("label")
    pq.write_table(table.set_column(index, table.schema.field(index), labels), output)
    with pytest.raises(ValueError, match="Canonical content SHA-256"):
        read_results(output)


def test_cutoff_ignores_late_corrections_future_matches_and_current_winner():
    cutoff = datetime(2030, 6, 10, 12, tzinfo=timezone.utc)
    prior = MatchResult(
        "prior",
        date(2030, 6, 9),
        cutoff - timedelta(hours=1),
        "Clay",
        "P02",
        "P03",
        "Robin",
        "New player",
        1,
    )
    current = replace(
        prior,
        match_id="F01",
        event_date=cutoff.date(),
        available_at=cutoff + timedelta(hours=12),
        p1_id="P01",
        p2_id="P02",
        p1_name="Alex",
        p2_name="Robin",
    )
    future = replace(
        current,
        match_id="future",
        event_date=date(2030, 6, 13),
        available_at=cutoff + timedelta(days=4),
    )
    correction = WinnerCorrection("prior", "P03", cutoff + timedelta(minutes=30))

    expected = {"P02": 1516.0, "P03": 1484.0}
    assert ratings_as_of([prior], cutoff) == expected
    assert ratings_as_of([future, current, prior], cutoff, [correction]) == expected
    assert ratings_as_of([prior, replace(current, label=0)], cutoff) == expected
    assert ratings_as_of([prior], correction.available_at, [correction]) == {
        "P02": 1484.0,
        "P03": 1516.0,
    }
    assert prior.label == 1
    earlier_correction = replace(correction, available_at=prior.available_at)
    with pytest.raises(ValueError, match="Correction must follow the original"):
        ratings_as_of([prior], cutoff, [earlier_correction])
    with pytest.raises(ValueError, match="Conflicting corrections"):
        ratings_as_of(
            [prior],
            correction.available_at,
            [correction, replace(correction, winner_id="P02")],
        )


def test_same_day_replay_uses_opening_ratings_and_waits_until_availability():
    event_date = date(2021, 5, 8)
    available = datetime(2021, 5, 10, tzinfo=timezone.utc)
    first = MatchResult(
        "a", event_date, available, "Hard", "P01", "P02", "First", "Second", 1
    )
    second = replace(first, match_id="b", p2_id="P03", p2_name="Third")
    assert ratings_as_of([first, second], available - timedelta(seconds=1)) == {}
    assert ratings_as_of([second, first], available) == {
        "P01": 1532.0,
        "P02": 1484.0,
        "P03": 1484.0,
    }
    for _, forecast in replay_matches([second, first]):
        assert forecast.cutoff == datetime(2021, 5, 7, 23, 59, 59, tzinfo=timezone.utc)
        assert forecast.p1_win_probability == 0.5


def test_worked_fixture_traces_noon_probability_and_later_correction():
    worked = worked_fixture()
    assert worked["fixture_id"] == "F01"
    assert worked["cutoff"] == "2030-06-10T12:00:00Z"
    assert (worked["p1_elo"], worked["p2_elo"]) == (1500.0, 1516.0)
    assert worked["p1_win_probability"] == pytest.approx(0.4769904127)
    assert worked["contributing_results"] == ["F01-prior"]
    assert worked["separate_correction"] == {
        "match_id": "F01-prior",
        "winner_id": "P03",
        "available_at": "2030-06-10T12:30:00+00:00",
    }


def test_leakage_comparison_uses_development_rows_only():
    scores = development_scores(normalize_results(SNAPSHOT.read_bytes()))
    assert scores["matches"] == 1487
    assert scores["as_of_log_loss"] == pytest.approx(0.674113138)
    assert scores["leaky_log_loss"] == pytest.approx(0.589333514)
