"""Acceptance tests for the strict training-serving feature contract."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from acebet.dataprep.dataprep import prepare_data
from acebet.app.dependencies.predict_winner import resolve_player_snapshots
from acebet.features import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    align_features_to_model,
    build_latest_player_stats,
    build_match_feature_row,
)
from acebet.train.train import (
    build_model,
    date_grouped_temporal_holdout,
    evaluate_model,
    get_lgb_params,
    prepare_data_for_training_clf,
)


@pytest.fixture(scope="module")
def fitted_contract_model():
    data = prepare_data_for_training_clf("2015-03-04", "2017-03-04")
    holdout = date_grouped_temporal_holdout(data)
    X_train = data.features.iloc[holdout.train_indices].copy()
    y_train = data.target[holdout.train_indices].copy()
    X_test = data.features.iloc[holdout.test_indices].copy()
    y_test = data.target[holdout.test_indices].copy()
    model = build_model(get_lgb_params()).fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


def test_pipeline_entrypoints_report_all_missing_contract_columns(
    fitted_contract_model,
):
    model, X_train, y_train, X_test, _ = fitted_contract_model
    missing_columns = ["best of", "surface"]
    incomplete_train = X_train.drop(columns=missing_columns)
    incomplete_test = X_test.drop(columns=missing_columns)

    with pytest.raises(ValueError, match="missing required columns") as fit_error:
        build_model(get_lgb_params()).fit(incomplete_train, y_train)
    with pytest.raises(ValueError, match="missing required columns") as predict_error:
        model.predict(incomplete_test)
    with pytest.raises(ValueError, match="missing required columns") as proba_error:
        model.predict_proba(incomplete_test)

    for error in (fit_error, predict_error, proba_error):
        assert all(column in str(error.value) for column in missing_columns)


def test_numeric_passthrough_preserves_holdout_values(fitted_contract_model):
    model, _, _, X_test, _ = fitted_contract_model
    preprocessor = model.named_steps["preprocessor"]

    transformed = preprocessor.transform(X_test)
    transformed_numeric = transformed.loc[:, NUMERIC_FEATURE_COLUMNS]

    pd.testing.assert_frame_equal(
        transformed_numeric.reset_index(drop=True),
        X_test.loc[:, NUMERIC_FEATURE_COLUMNS].reset_index(drop=True),
    )
    assert not transformed_numeric.isna().any().any()
    assert all(
        isinstance(transformed[column].dtype, pd.CategoricalDtype)
        for column in CATEGORICAL_FEATURE_COLUMNS
    )


def test_preprocessor_and_lightgbm_receive_only_the_contract(
    fitted_contract_model,
):
    model, _, _, _, _ = fitted_contract_model
    preprocessor = model.named_steps["preprocessor"]
    gbm = model.named_steps["gbm"]

    assert list(model.feature_names_in_) == list(MODEL_FEATURE_COLUMNS)
    assert list(preprocessor.feature_names_in_) == list(MODEL_FEATURE_COLUMNS)
    assert list(preprocessor.get_feature_names_out()) == list(MODEL_FEATURE_COLUMNS)
    assert preprocessor.output_indices_["numeric"] == slice(0, 10)
    assert preprocessor.output_indices_["categorical"] == slice(10, 17)
    assert gbm.booster_.params["categorical_column"] == list(range(10, 17))
    assert len(gbm.booster_.pandas_categorical) == len(CATEGORICAL_FEATURE_COLUMNS)

    removed_raw_columns = {
        "target",
        "date",
        "atp",
        "location",
        "tournament",
        "sets_p1",
        "sets_p2",
        "b365_p1",
        "b365_p2",
        "ps_p1",
        "ps_p2",
        "comment",
    }
    assert removed_raw_columns.isdisjoint(preprocessor.get_feature_names_out())


def test_all_unseen_categorical_values_score_successfully(fitted_contract_model):
    model, _, _, X_test, _ = fitted_contract_model
    scoring_row = X_test.head(1).copy()
    for column in CATEGORICAL_FEATURE_COLUMNS:
        scoring_row[column] = f"never-seen-{column}"

    prediction = model.predict(scoring_row)
    probabilities = model.predict_proba(scoring_row)

    assert prediction.shape == (1,)
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_training_online_model_and_metadata_share_one_contract(
    fitted_contract_model,
):
    model, X_train, _, _, _ = fitted_contract_model
    prepared = prepare_data()
    player_stats = build_latest_player_stats(prepared)
    p1_snapshot, p2_snapshot = resolve_player_snapshots(
        player_stats, "Fognini F.", "Jarry N."
    )
    online_features = build_match_feature_row(
        p1_name="Fognini F.",
        p2_name="Jarry N.",
        date="2030-01-01",
        p1_snapshot=p1_snapshot.feature_values(),
        p2_snapshot=p2_snapshot.feature_values(),
        series="ATP250",
        court="Outdoor",
        surface="Hard",
        round="1st Round",
        best_of=3,
    )
    aligned_online = align_features_to_model(model, online_features)

    assert list(X_train.columns) == list(MODEL_FEATURE_COLUMNS)
    assert list(online_features.columns) == list(MODEL_FEATURE_COLUMNS)
    assert list(aligned_online.columns) == list(MODEL_FEATURE_COLUMNS)
    assert list(model.feature_names_in_) == list(MODEL_FEATURE_COLUMNS)


def test_all_metrics_are_finite_and_in_valid_ranges(fitted_contract_model):
    model, _, _, X_test, y_test = fitted_contract_model

    metrics = evaluate_model(model, X_test, y_test)

    assert set(metrics) == {"accuracy", "log_loss", "brier_score", "roc_auc"}
    assert all(math.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
