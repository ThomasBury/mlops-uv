import math

import joblib
import numpy as np
import pandas as pd
import pytest

from acebet.features import MODEL_FEATURE_COLUMNS
from acebet.model_artifacts import compute_selection
from acebet.train import train as train_module
from acebet.train.train import EloBaselineClassifier, train_model


def feature_frame(probabilities):
    rows = len(probabilities)
    frame = pd.DataFrame(index=range(rows), columns=MODEL_FEATURE_COLUMNS)
    for column in MODEL_FEATURE_COLUMNS:
        frame[column] = 0.0
    frame["proba_elo"] = probabilities
    return frame


def candidates(constant, elo, lightgbm, *, elo_accuracy=0.5, lightgbm_accuracy=0.5):
    def metrics(loss, accuracy):
        return {
            "accuracy": accuracy,
            "log_loss": loss,
            "brier_score": 0.25,
            "roc_auc": 0.5,
        }

    return {
        "constant": {"metrics": metrics(constant, 0.5)},
        "elo": {"metrics": metrics(elo, elo_accuracy)},
        "lightgbm": {"metrics": metrics(lightgbm, lightgbm_accuracy)},
    }


def test_elo_baseline_is_sklearn_compatible_and_round_trips(tmp_path):
    X = feature_frame([0.2, 0.8, 0.5, 0.7])
    model = EloBaselineClassifier().fit(X, np.array([0, 1, 0, 1]))

    assert list(model.classes_) == [0, 1]
    assert model.n_features_in_ == len(MODEL_FEATURE_COLUMNS)
    assert list(model.feature_names_in_) == list(MODEL_FEATURE_COLUMNS)
    np.testing.assert_allclose(
        model.predict_proba(X), [[0.8, 0.2], [0.2, 0.8], [0.5, 0.5], [0.3, 0.7]]
    )
    np.testing.assert_array_equal(model.predict(X), [0, 1, 0, 1])
    path = tmp_path / "elo.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))


@pytest.mark.parametrize(
    "values, message",
    [
        ([np.nan], "finite"),
        ([np.inf], "finite"),
        ([-0.1], "in \\[0, 1\\]"),
        ([1.1], "in \\[0, 1\\]"),
    ],
)
def test_elo_baseline_rejects_invalid_probabilities(values, message):
    X_train = feature_frame([0.2, 0.8])
    model = EloBaselineClassifier().fit(X_train, np.array([0, 1]))
    with pytest.raises(ValueError, match=message):
        model.predict_proba(feature_frame(values))


def test_selection_covers_challenger_elo_and_sole_servable_outcomes():
    assert (
        compute_selection(candidates(0.70, 0.65, 0.64))["reason_code"]
        == "lightgbm_beats_elo_margin"
    )
    assert (
        compute_selection(candidates(0.70, 0.65, 0.646))["reason_code"]
        == "elo_preferred_challenger_margin_not_met"
    )
    assert (
        compute_selection(candidates(0.70, 0.69, 0.699))["reason_code"]
        == "elo_only_servable"
    )
    assert (
        compute_selection(candidates(0.70, 0.699, 0.69))["reason_code"]
        == "lightgbm_only_servable"
    )


def test_exact_margins_pass_and_secondary_metrics_do_not_override_log_loss():
    at_constant_margin = compute_selection(
        candidates(0.700, 0.695, 0.700, elo_accuracy=0.0, lightgbm_accuracy=1.0)
    )
    assert at_constant_margin["selected_model_kind"] == "elo_baseline"
    at_challenger_margin = compute_selection(
        candidates(0.700, 0.690, 0.685, elo_accuracy=1.0, lightgbm_accuracy=0.0)
    )
    assert at_challenger_margin["selected_model_kind"] == "lightgbm"


def test_no_candidate_beats_constant_has_stable_failure_reason():
    selection = compute_selection(candidates(0.700, 0.696, 0.6951))
    assert selection["status"] == "failed"
    assert selection["selected_model_kind"] is None
    assert selection["reason_code"] == "no_candidate_beats_constant"


def test_canonical_training_selects_elo_with_expected_metrics(tmp_path):
    result = train_model(
        "2015-03-04", "2017-03-04", output_dir=tmp_path, export_joblib=True
    )

    assert result.model_kind == "elo_baseline"
    assert math.isclose(
        result.candidate_metrics["constant"]["log_loss"], 0.69315, abs_tol=5e-6
    )
    assert math.isclose(
        result.candidate_metrics["elo"]["log_loss"], 0.60210, abs_tol=5e-6
    )
    assert math.isclose(
        result.candidate_metrics["lightgbm"]["log_loss"], 0.61809, abs_tol=5e-6
    )
    assert result.metrics == result.candidate_metrics["elo"]
    assert result.selection["reason_code"] == "elo_preferred_challenger_margin_not_met"
    assert result.exported_model_path is not None


def test_total_gate_failure_writes_no_bundle(tmp_path, monkeypatch):
    failed_metrics = {
        "accuracy": 1.0,
        "log_loss": 0.69,
        "brier_score": 0.1,
        "roc_auc": 1.0,
    }
    monkeypatch.setattr(
        train_module, "evaluate_model", lambda model, X_test, y_test: failed_metrics
    )

    with pytest.raises(ValueError, match="no_candidate_beats_constant"):
        train_model(
            "2015-03-04",
            "2017-03-04",
            output_dir=tmp_path,
            export_joblib=True,
        )

    assert list(tmp_path.iterdir()) == []
