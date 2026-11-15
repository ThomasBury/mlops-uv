import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from acebet.dataprep.dataprep import prepare_data
from acebet.train import train as train_module
from acebet.train.train import (
    PreparedTrainingData,
    date_grouped_temporal_holdout,
    prepare_data_for_training_clf,
    train_model,
)


def _prepared_frame(dates: list[str], targets: list[int]) -> pd.DataFrame:
    row_count = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "p1": [f"Player {index}A" for index in range(row_count)],
            "p2": [f"Player {index}B" for index in range(row_count)],
            "rank_p1": np.arange(1, row_count + 1),
            "rank_p2": np.arange(2, row_count + 2),
            "elo_p1": np.linspace(1500.0, 1600.0, row_count),
            "elo_p2": np.linspace(1450.0, 1550.0, row_count),
            "proba_elo": np.full(row_count, 0.5),
            "target": targets,
        }
    )


def _write_prepared_data(tmp_path, dates: list[str], targets: list[int]) -> Path:
    frame = _prepared_frame(dates, targets)
    path = tmp_path / "prepared.feather"
    frame.to_feather(path)
    return path


def _holdout_input(dates: list[object], target: list[int]) -> PreparedTrainingData:
    return PreparedTrainingData(
        features=pd.DataFrame({"feature": np.arange(len(target))}),
        target=np.asarray(target),
        dates=pd.Series(dates, name="date"),
    )


def test_prepare_data_stably_sorts_normalized_calendar_dates(tmp_path):
    path = _write_prepared_data(
        tmp_path,
        [
            "2024-01-02 08:00:00",
            "2024-01-01 12:00:00",
            "2024-01-01 08:00:00",
            "2024-01-03 08:00:00",
        ],
        [0, 1, 0, 1],
    )

    first = prepare_data(path)
    second = prepare_data(path)

    assert list(first["p1"]) == ["Player 1A", "Player 2A", "Player 0A", "Player 3A"]
    assert first["date"].dt.hour.eq(0).all()
    pd.testing.assert_frame_equal(first, second)


def test_date_grouped_temporal_holdout_moves_cutoff_to_start_of_date():
    data = _holdout_input(
        [
            "2024-01-01",
            "2024-01-01",
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
            "2024-01-03",
            "2024-01-04",
            "2024-01-04",
        ],
        [0, 1, 0, 1, 0, 1, 0, 0, 1],
    )

    holdout = date_grouped_temporal_holdout(data)
    dates = pd.to_datetime(data.dates)
    train_dates = dates.iloc[holdout.train_indices]
    test_dates = dates.iloc[holdout.test_indices]

    assert holdout.cutoff_date == pd.Timestamp("2024-01-03")
    np.testing.assert_array_equal(holdout.train_indices, np.arange(4))
    np.testing.assert_array_equal(holdout.test_indices, np.arange(4, 9))
    assert train_dates.max() < test_dates.min()
    assert set(train_dates).isdisjoint(set(test_dates))


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            PreparedTrainingData(
                features=pd.DataFrame({"feature": range(5)}),
                target=np.array([0, 1, 0, 1, 0, 1]),
                dates=pd.Series(pd.date_range("2024-01-01", periods=6)),
            ),
            "misaligned",
        ),
        (
            _holdout_input(
                ["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04"],
                [0, 1, 0, 1],
            ),
            "monotonically increasing",
        ),
        (
            _holdout_input(
                ["2024-01-01", pd.NaT, "2024-01-03", "2024-01-04"],
                [0, 1, 0, 1],
            ),
            "must not contain null",
        ),
        (
            _holdout_input(
                ["2024-01-01", "2024-01-02", "2024-01-03"],
                [0, 1, 0],
            ),
            "require at least 4",
        ),
        (
            _holdout_input(["2024-01-01"] * 4, [0, 1, 0, 1]),
            "at least two distinct calendar dates",
        ),
        (
            _holdout_input(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                [1, 1, 1, 1],
            ),
            "Training window must contain both target classes",
        ),
        (
            _holdout_input(
                ["2024-01-01"] * 5 + ["2024-01-02"],
                [0, 1, 0, 1, 0, 1],
            ),
            "empty training partition",
        ),
        (
            _holdout_input(
                pd.date_range("2024-01-01", periods=6).astype(str).tolist(),
                [0, 0, 0, 0, 0, 1],
            ),
            "Training partition must contain both target classes",
        ),
        (
            _holdout_input(
                pd.date_range("2024-01-01", periods=6).astype(str).tolist(),
                [0, 1, 0, 1, 1, 1],
            ),
            "Testing partition must contain both target classes",
        ),
    ],
)
def test_date_grouped_temporal_holdout_rejects_invalid_inputs(data, message):
    with pytest.raises(ValueError, match=message):
        date_grouped_temporal_holdout(data)


def test_training_window_rejects_invalid_empty_and_reversed_boundaries(
    tmp_path, monkeypatch
):
    path = _write_prepared_data(
        tmp_path,
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        [0, 1, 0, 1],
    )
    monkeypatch.setattr(train_module, "DEFAULT_DATA_PATH", path)

    with pytest.raises(ValueError, match="start_date must be a parseable date"):
        prepare_data_for_training_clf("not-a-date", "2024-01-04")
    with pytest.raises(ValueError, match="Training window is reversed"):
        prepare_data_for_training_clf("2024-01-04", "2024-01-01")
    with pytest.raises(ValueError, match="contains no rows"):
        prepare_data_for_training_clf("2025-01-01", "2025-01-02")


@pytest.mark.parametrize(
    ("dates", "targets", "message"),
    [
        (
            ["2024-01-01", "2024-01-02", "2024-01-03"],
            [0, 1, 0],
            "require at least 4",
        ),
        (
            ["2024-01-01"] * 4,
            [0, 1, 0, 1],
            "at least two distinct calendar dates",
        ),
        (
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            [0, 0, 0, 0],
            "Training window must contain both target classes",
        ),
    ],
)
def test_training_window_rejects_too_small_and_single_class_data(
    tmp_path, monkeypatch, dates, targets, message
):
    path = _write_prepared_data(tmp_path, dates, targets)
    monkeypatch.setattr(train_module, "DEFAULT_DATA_PATH", path)

    with pytest.raises(ValueError, match=message):
        prepare_data_for_training_clf(min(dates), max(dates))


def test_repeated_preparation_holdout_and_training_are_deterministic():
    first = prepare_data_for_training_clf("2015-03-04", "2017-03-04")
    second = prepare_data_for_training_clf("2015-03-04", "2017-03-04")

    pd.testing.assert_frame_equal(first.features, second.features)
    pd.testing.assert_series_equal(first.dates, second.dates)
    np.testing.assert_array_equal(first.target, second.target)

    first_holdout = date_grouped_temporal_holdout(first)
    second_holdout = date_grouped_temporal_holdout(second)
    assert first_holdout.cutoff_date == second_holdout.cutoff_date
    np.testing.assert_array_equal(
        first_holdout.train_indices, second_holdout.train_indices
    )
    np.testing.assert_array_equal(
        first_holdout.test_indices, second_holdout.test_indices
    )

    first_result = train_model("2015-03-04", "2017-03-04")
    second_result = train_model("2015-03-04", "2017-03-04")
    assert first_result.metrics == second_result.metrics
