from __future__ import annotations

from math import log
import sys
from pathlib import Path

import pandas as pd
import pytest

from acebet.dataprep.dataprep import prepare_data, summarize_elo_causal_evidence
from acebet.dataprep.raw_to_prepared import (
    PREPARED_COLUMN_ORDER,
    REQUIRED_RAW_COLUMNS,
    build_prepared_dataset,
    normalize_raw_matches,
    parse_args,
)
from acebet.features import elo_win_probability

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _raw_rows() -> list[dict[str, object]]:
    return [
        {
            "date": "2024-01-03",
            "p1": "Player A",
            "p2": "Player B",
            "rank_p1": 10,
            "rank_p2": 20,
            "sets_p1": 2,
            "sets_p2": 1,
            "elo_p1": 1700.0,
            "elo_p2": 1600.0,
            "surface": "Clay",
            "round": "Quarterfinals",
            "comment": "Completed",
        },
        {
            "date": "2024-01-01",
            "p1": "Player C",
            "p2": "Player D",
            "rank_p1": 30,
            "rank_p2": 15,
            "sets_p1": 0,
            "sets_p2": 2,
            "elo_p1": 1500.0,
            "elo_p2": 1550.0,
        },
    ]


def _valid_prepared_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "p1": "Player A",
                "p2": "Player B",
                "rank_p1": 10,
                "rank_p2": 20,
                "elo_p1": 1700.0,
                "elo_p2": 1600.0,
                "proba_elo": elo_win_probability(1700.0, 1600.0),
                "target": True,
            }
        ]
    )


def _write_prepared(tmp_path: Path, df: pd.DataFrame) -> Path:
    input_path = tmp_path / "prepared.feather"
    df.to_feather(input_path)
    return input_path


def test_build_prepared_dataset_rejects_missing_required_raw_columns(tmp_path: Path):
    input_path = tmp_path / "matches.csv"
    pd.DataFrame([{"date": "2024-01-01", "p1": "A", "p2": "B"}]).to_csv(
        input_path, index=False
    )

    with pytest.raises(ValueError, match="missing required columns"):
        build_prepared_dataset(input_path, tmp_path / "prepared.feather")


@pytest.mark.parametrize("column", REQUIRED_RAW_COLUMNS)
def test_build_prepared_dataset_rejects_null_raw_values(tmp_path, column):
    raw = pd.DataFrame(_raw_rows())
    raw[column] = raw[column].astype(object)
    raw.loc[0, column] = None
    input_path = tmp_path / "matches.csv"
    output_path = tmp_path / "prepared.feather"
    raw.to_csv(input_path, index=False)

    with pytest.raises(ValueError, match=f"null values in required columns.*{column}"):
        build_prepared_dataset(input_path, output_path)

    assert not output_path.exists()


@pytest.mark.parametrize("existing_output", [False, True])
def test_failed_build_preserves_destination(tmp_path, existing_output):
    input_path = tmp_path / "matches.csv"
    output_path = tmp_path / "prepared.feather"
    if existing_output:
        _valid_prepared_df().to_feather(output_path)
        original = output_path.read_bytes()
    raw = pd.DataFrame(_raw_rows())
    raw.loc[0, "rank_p1"] = 0
    raw.to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="rank columns must be positive"):
        build_prepared_dataset(input_path, output_path)

    if existing_output:
        assert output_path.read_bytes() == original
    else:
        assert not output_path.exists()


def test_prepare_data_rejects_null_required_values(tmp_path: Path):
    prepared = _valid_prepared_df()
    prepared.loc[0, "p1"] = None

    with pytest.raises(ValueError, match="null values in required columns"):
        prepare_data(_write_prepared(tmp_path, prepared))


def test_prepare_data_rejects_invalid_dates(tmp_path: Path):
    prepared = _valid_prepared_df()
    prepared.loc[0, "date"] = "not-a-date"

    with pytest.raises(ValueError, match="unparseable date"):
        prepare_data(_write_prepared(tmp_path, prepared))


def test_prepare_data_rejects_out_of_range_elo_probability(tmp_path: Path):
    prepared = _valid_prepared_df()
    prepared.loc[0, "proba_elo"] = 1.2

    with pytest.raises(ValueError, match="proba_elo must be within"):
        prepare_data(_write_prepared(tmp_path, prepared))


def test_prepare_data_rejects_invalid_targets(tmp_path: Path):
    prepared = _valid_prepared_df()
    prepared["target"] = [2]

    with pytest.raises(ValueError, match="target must be boolean or binary"):
        prepare_data(_write_prepared(tmp_path, prepared))


def test_prepare_data_rejects_non_positive_ranks(tmp_path: Path):
    prepared = _valid_prepared_df()
    prepared.loc[0, "rank_p1"] = 0

    with pytest.raises(ValueError, match="rank columns must be positive"):
        prepare_data(_write_prepared(tmp_path, prepared))


def test_build_prepared_dataset_fills_optional_columns_and_derives_features(
    tmp_path: Path,
):
    input_path = tmp_path / "matches.csv"
    output_path = tmp_path / "prepared.feather"
    pd.DataFrame(_raw_rows()).to_csv(input_path, index=False)

    prepared = build_prepared_dataset(input_path, output_path)

    assert list(prepared.columns) == list(PREPARED_COLUMN_ORDER)
    assert prepared["date"].is_monotonic_increasing
    assert prepared.loc[0, "location"] == "Unknown"
    assert prepared.loc[0, "tournament"] == "Online Match"
    assert prepared.loc[0, "series"] == "ATP250"
    assert prepared.loc[0, "court"] == "Outdoor"
    assert prepared.loc[0, "surface"] == "Hard"
    assert prepared.loc[0, "round"] == "1st Round"
    assert prepared.loc[0, "best of"] == 3
    assert pd.isna(prepared.loc[0, "ps_p1"])
    assert pd.isna(prepared.loc[0, "b365_p2"])
    assert bool(prepared.loc[0, "target"]) is False
    assert prepared.loc[0, "year"] == 2024
    assert prepared.loc[0, "month"] == 1
    assert prepared.loc[0, "day"] == 1
    assert prepared.loc[0, "rank_diff"] == 15
    assert prepared.loc[0, "best_ranked"] == "p2"

    assert prepared.loc[1, "surface"] == "Clay"
    assert prepared.loc[1, "round"] == "Quarterfinals"
    assert prepared.loc[1, "comment"] == "Completed"
    assert bool(prepared.loc[1, "target"]) is True
    assert prepared.loc[1, "proba_elo"] == pytest.approx(
        elo_win_probability(1700.0, 1600.0)
    )


def test_build_prepared_dataset_supports_directory_inputs(tmp_path: Path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    output_path = tmp_path / "prepared.feather"
    pd.DataFrame([_raw_rows()[0]]).to_csv(input_dir / "b_matches.csv", index=False)
    pd.DataFrame([_raw_rows()[1]]).to_csv(input_dir / "a_matches.csv", index=False)

    prepared = build_prepared_dataset(input_dir, output_path)

    assert list(prepared["p1"]) == ["Player C", "Player A"]
    validated = prepare_data(output_path)
    assert list(validated.columns) == list(PREPARED_COLUMN_ORDER)


def test_normalize_raw_matches_stably_orders_same_day_rows():
    rows = _raw_rows()
    same_day_rows = [
        {**rows[0], "date": "2024-01-02 12:00:00", "p1": "First"},
        {**rows[1], "date": "2024-01-01 12:00:00", "p1": "Second"},
        {**rows[0], "date": "2024-01-01 08:00:00", "p1": "Third"},
    ]

    normalized = normalize_raw_matches(pd.DataFrame(same_day_rows))

    assert list(normalized["p1"]) == ["Second", "Third", "First"]
    assert normalized["date"].dt.hour.eq(0).all()


def test_raw_builder_cli_requires_explicit_output_path(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["raw_to_prepared", "--input-path", "raw.csv"],
    )

    with pytest.raises(SystemExit):
        parse_args()


def test_committed_raw_sample_rebuild_matches_packaged_sample_schema_and_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    raw_sample_path = REPOSITORY_ROOT / "data" / "raw" / "atp_matches_sample.csv"
    output_path = tmp_path / "sample.feather"
    monkeypatch.chdir(tmp_path)

    rebuilt = build_prepared_dataset(raw_sample_path, output_path)
    packaged = pd.read_feather(
        REPOSITORY_ROOT / "src" / "acebet" / "data" / "atp_data_sample.feather"
    )

    assert list(rebuilt.columns) == list(packaged.columns)
    assert len(rebuilt) == len(packaged) == 100
    pd.testing.assert_series_equal(
        pd.util.hash_pandas_object(rebuilt, index=False)
        .sort_values(ignore_index=True)
        .rename(None),
        pd.util.hash_pandas_object(packaged, index=False)
        .sort_values(ignore_index=True)
        .rename(None),
    )


@pytest.mark.parametrize(
    "column", ["rank_p1", "rank_p2", "elo_p1", "elo_p2", "proba_elo"]
)
@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_prepared_rejects_nonfinite_numbers(tmp_path, column, value):
    prepared = _valid_prepared_df().astype({column: float})
    prepared.loc[0, column] = value
    with pytest.raises(ValueError, match="finite"):
        prepare_data(_write_prepared(tmp_path, prepared))


@pytest.mark.parametrize("column", ["rank_p1", "rank_p2"])
def test_prepared_rejects_fractional_ranks(tmp_path, column):
    prepared = _valid_prepared_df().astype({column: float})
    prepared.loc[0, column] = 1.5
    with pytest.raises(ValueError, match="integral"):
        prepare_data(_write_prepared(tmp_path, prepared))


@pytest.mark.parametrize(
    "column", ["rank_p1", "rank_p2", "elo_p1", "elo_p2", "sets_p1", "sets_p2"]
)
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), "invalid"])
def test_raw_rejects_bad_numbers_before_feature_calculation(
    tmp_path, monkeypatch, column, value
):
    import acebet.dataprep.raw_to_prepared as raw_module

    def must_not_derive(*_args, **_kwargs):
        pytest.fail("Invalid inputs reached feature calculations")

    monkeypatch.setattr(raw_module, "elo_win_probability", must_not_derive)
    monkeypatch.setattr(raw_module, "add_derived_match_features", must_not_derive)
    raw = pd.DataFrame(_raw_rows()).astype({column: object})
    raw.loc[0, column] = value
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "prepared.feather"
    raw.to_csv(input_path, index=False)
    with pytest.raises(ValueError, match="finite|non-numeric"):
        build_prepared_dataset(input_path, output_path)
    assert not output_path.exists()


@pytest.mark.parametrize("column", ["rank_p1", "rank_p2"])
def test_raw_rejects_fractional_ranks_before_integer_conversion(tmp_path, column):
    raw = pd.DataFrame(_raw_rows()).astype({column: float})
    raw.loc[0, column] = 1.5
    input_path = tmp_path / "raw.csv"
    raw.to_csv(input_path, index=False)
    output_path = tmp_path / "prepared.feather"
    with pytest.raises(ValueError, match="integral"):
        build_prepared_dataset(input_path, output_path)
    assert not output_path.exists()


def test_prepared_numeric_strings_become_numeric_columns(tmp_path):
    from acebet.dataprep.dataprep import NUMERIC_CONTRACT_COLUMNS

    prepared = _valid_prepared_df()
    for column in NUMERIC_CONTRACT_COLUMNS:
        prepared[column] = prepared[column].astype(str)
    result = prepare_data(_write_prepared(tmp_path, prepared))
    assert all(
        pd.api.types.is_numeric_dtype(result[column])
        for column in NUMERIC_CONTRACT_COLUMNS
    )
    assert result.loc[0, "rank_diff"] == -10


@pytest.mark.parametrize("sets", [0, 1, 2])
def test_raw_equal_set_counts_reject_without_replacing_output(tmp_path, sets):
    raw = pd.DataFrame(_raw_rows())
    raw.loc[0, ["sets_p1", "sets_p2"]] = sets
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "prepared.feather"
    raw.to_csv(input_path, index=False)
    output_path.write_bytes(b"previous output")
    with pytest.raises(ValueError, match="equal set counts have no winner"):
        build_prepared_dataset(input_path, output_path)
    assert output_path.read_bytes() == b"previous output"


@pytest.mark.parametrize("column", ["elo_p1", "elo_p2"])
@pytest.mark.parametrize("bad_elo", [0.0, -100.0, 5000.1])
def test_prepare_data_rejects_out_of_bounds_elo_ratings(tmp_path, column, bad_elo):
    prepared = _valid_prepared_df()
    prepared.loc[0, column] = bad_elo
    with pytest.raises(ValueError, match="Elo columns must be within"):
        prepare_data(_write_prepared(tmp_path, prepared))


def test_prepare_data_rejects_inconsistent_proba_elo(tmp_path):
    prepared = _valid_prepared_df()
    prepared.loc[0, "proba_elo"] = 0.25
    with pytest.raises(
        ValueError, match="proba_elo must match the Elo win probability formula"
    ):
        prepare_data(_write_prepared(tmp_path, prepared))


def test_prepare_data_accepts_unsigned_elo_ratings(tmp_path):
    prepared = _valid_prepared_df().astype({"elo_p1": "uint16", "elo_p2": "uint16"})

    result = prepare_data(_write_prepared(tmp_path, prepared))

    assert result.loc[0, "proba_elo"] == prepared.loc[0, "proba_elo"]


def test_elo_win_probability_handles_extreme_rating_gaps_without_overflow():
    assert elo_win_probability(0.0, 1_000_000.0) == 0.0
    assert elo_win_probability(1_000_000.0, 0.0) == 1.0


def test_summarize_elo_causal_evidence_verifies_transitions_and_odds_alignment():
    df = prepare_data()
    evidence = summarize_elo_causal_evidence(df)

    assert evidence["player_transitions"] == 75_072
    assert evidence["post_win_elo_increase_rate"] > 0.99
    assert evidence["post_loss_elo_decrease_rate"] > 0.99
    assert evidence["elo_increase_by_next_win_rate"] < 0.55
    assert evidence["elo_decrease_by_next_loss_rate"] < 0.55
    assert evidence["pinnacle_odds_rows"] > 30_000
    assert (
        evidence["pinnacle_log_loss_on_odds_subset"]
        < evidence["elo_log_loss_on_odds_subset"]
    )
    assert evidence["elo_pinnacle_probability_correlation"] > 0.85


@pytest.mark.parametrize("valid_rows", [1, 2])
def test_elo_probe_counts_and_scores_single_class_odds(valid_rows):
    valid = _valid_prepared_df().assign(ps_p1=1.5, ps_p2=3.0)
    matches = pd.concat(
        [valid] * valid_rows + [valid.assign(ps_p1=1.0)], ignore_index=True
    )

    evidence = summarize_elo_causal_evidence(matches)

    assert evidence["pinnacle_odds_rows"] == valid_rows
    assert evidence["elo_log_loss_on_odds_subset"] == pytest.approx(
        -log(elo_win_probability(1700.0, 1600.0))
    )
    assert evidence["pinnacle_log_loss_on_odds_subset"] == pytest.approx(-log(2 / 3))
    assert "elo_pinnacle_probability_correlation" not in evidence
