import pytest

from acebet.forecast import probability_for_first_player


def test_probability_follows_requested_player_order():
    assert probability_for_first_player("P01", "P02", 0.70) == 0.70
    assert probability_for_first_player("P02", "P01", 0.70) == pytest.approx(0.30)
    assert probability_for_first_player("P02", "P01", 0.0) == 1.0


@pytest.mark.parametrize(
    ("first", "second", "probability"),
    [("", "P02", 0.5), ("P01", "P01", 0.5), ("P01", "P02", float("nan"))],
)
def test_invalid_forecast_inputs_are_rejected(first, second, probability):
    with pytest.raises(ValueError):
        probability_for_first_player(first, second, probability)
