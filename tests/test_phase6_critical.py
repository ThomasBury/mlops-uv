"""Phase 6 branch coverage for artifact and prediction invariants."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
import shutil
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from acebet.app.dependencies import predict_winner
from acebet.app.dependencies.predict_winner import (
    PlayerSnapshot,
    UnknownPlayerSnapshotsError,
    calculate_feature_freshness,
    load_model_bundle,
    load_player_stats,
    normalize_prediction_outputs,
    resolve_player_snapshots,
)
from acebet import model_artifacts
from acebet.model_artifacts import (
    candidates_with_selection_details,
    compute_selection,
    ensure_artifact_bundle,
    list_exported_models,
    model_card_path,
    model_metadata_path,
    promotion_record_path,
    read_json,
    resolve_latest_model_path,
    resolve_model_artifact_path,
    validate_artifact_bundle,
    validate_model_metadata,
    validate_promotion_record,
    write_json,
)

PACKAGE_DATA = Path(model_artifacts.__file__).parent / "data"
MODEL_NAME = "model_2026-08-02-12-59-07.joblib"


def canonical_metadata() -> dict:
    return read_json(PACKAGE_DATA / MODEL_NAME.replace(".joblib", ".metadata.json"))


def copy_bundle(destination: Path) -> Path:
    for name in (
        MODEL_NAME,
        MODEL_NAME.replace(".joblib", ".metadata.json"),
        MODEL_NAME.replace(".joblib", ".modelcard.md"),
        "promoted_model.json",
    ):
        shutil.copy2(PACKAGE_DATA / name, destination / name)
    return destination / MODEL_NAME


def set_nested(payload: dict, path: str, value: object) -> None:
    target = payload
    keys = path.split(".")
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("dataset", None, "dataset section"),
        ("dataset.source_path", "/absolute/data.feather", "source_path"),
        ("dataset.source_sha256", "INVALID", "source_sha256"),
        ("dataset.row_counts.total", 1, "row_counts"),
        ("dataset.requested_window", None, "requested_window"),
        ("dataset.holdout_cutoff", 1, "holdout_cutoff"),
        ("feature_contract", {}, "feature contract"),
        ("candidates", [], "candidates section"),
        ("candidates.constant.metrics", {"accuracy": 0.5}, "metrics must contain"),
        ("candidates.constant.metrics.accuracy", float("nan"), "finite and numeric"),
        ("candidates.constant.metrics.accuracy", 2.0, "accuracy or log_loss"),
        ("candidates.constant.metrics.brier_score", 2.0, "brier_score"),
        ("candidates.training_prevalence", True, "Training prevalence"),
        (
            "candidates.elo.log_loss_improvement_over_constant",
            123.0,
            "improvement over constant",
        ),
        ("candidates.elo.servable", False, "servability flag"),
        ("candidates.lightgbm.parameters", [], "parameters are missing"),
        ("selection", {}, "Recorded model selection"),
        ("model.kind", "lightgbm", "Model kind"),
        ("model.package.version", "999", "package version mismatch"),
        ("model.serialization", "pickle", "trusted-only joblib"),
        ("model.sha256", "bad", "Model sha256"),
    ],
)
def test_invalid_model_metadata_is_rejected(path, value, message):
    metadata = canonical_metadata()
    set_nested(metadata, path, value)
    with pytest.raises(ValueError, match=message):
        validate_model_metadata(metadata)


def test_missing_metadata_section_and_unsupported_schema_are_rejected():
    metadata = canonical_metadata()
    del metadata["model"]
    with pytest.raises(ValueError, match="missing sections"):
        validate_model_metadata(metadata)

    metadata = canonical_metadata()
    metadata["schema_version"] = 999
    with pytest.raises(ValueError, match="Unsupported model metadata schema"):
        validate_model_metadata(metadata)


@pytest.mark.parametrize("loss", [True, float("inf"), -1.0])
def test_selection_rejects_invalid_log_loss(loss):
    candidates = canonical_metadata()["candidates"]
    candidates["elo"]["metrics"]["log_loss"] = loss
    message = "numeric" if loss is True else "finite and non-negative"
    with pytest.raises(ValueError, match=message):
        compute_selection(candidates)


def test_metadata_rejects_selection_with_no_servable_candidate():
    metadata = canonical_metadata()
    metrics = {
        name: deepcopy(metadata["candidates"][name]["metrics"])
        for name in ("constant", "elo", "lightgbm")
    }
    metrics["constant"]["log_loss"] = 0.5
    metrics["elo"]["log_loss"] = 0.499
    metrics["lightgbm"]["log_loss"] = 0.498
    metadata["candidates"] = candidates_with_selection_details(
        metrics, training_prevalence=0.5, lightgbm_parameters={"n_estimators": 1}
    )
    metadata["selection"] = compute_selection(metadata["candidates"])
    with pytest.raises(ValueError, match="no servable selected candidate"):
        validate_model_metadata(metadata)


def test_json_and_artifact_listing_helpers(tmp_path):
    array_path = tmp_path / "array.json"
    array_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON object"):
        read_json(array_path)

    older = tmp_path / "model_2024.joblib"
    newer = tmp_path / "model_2025.joblib"
    newer.write_bytes(b"new")
    older.write_bytes(b"old")
    assert list_exported_models(tmp_path) == [older, newer]
    assert resolve_latest_model_path(tmp_path) == newer
    for path in (older, newer):
        path.unlink()
    with pytest.raises(FileNotFoundError, match="No trained model"):
        resolve_latest_model_path(tmp_path)


def test_direct_bundle_validates_filename_and_checksum(tmp_path):
    model_path = copy_bundle(tmp_path)
    metadata_path = model_metadata_path(model_path)
    metadata = read_json(metadata_path)
    metadata["model"]["filename"] = "wrong.joblib"
    write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="filename"):
        validate_artifact_bundle(model_path)

    copy_bundle(tmp_path)
    model_path.write_bytes(model_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_artifact_bundle(model_path)


def test_incomplete_bundle_reports_every_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        ensure_artifact_bundle(tmp_path / "model_missing.joblib")


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("unsafe_name", ValueError),
        ("missing_file", FileNotFoundError),
        ("sidecar_name", ValueError),
        ("identity", ValueError),
    ],
)
def test_promotion_record_validation_defenses(tmp_path, mutation, error):
    model_path = copy_bundle(tmp_path)
    record = read_json(promotion_record_path(tmp_path))
    if mutation == "unsafe_name":
        record["model_filename"] = "../model.joblib"
    elif mutation == "missing_file":
        model_card_path(model_path).unlink()
    elif mutation == "sidecar_name":
        record["metadata_filename"] = record["modelcard_filename"]
    else:
        record["model_kind"] = "lightgbm"
    with pytest.raises(error):
        validate_promotion_record(record, tmp_path)


def test_resolve_direct_model_ignores_pointer_to_another_bundle(tmp_path):
    model_path = copy_bundle(tmp_path)
    record = read_json(promotion_record_path(tmp_path))
    record["model_filename"] = "model_other.joblib"
    write_json(promotion_record_path(tmp_path), record)
    assert resolve_model_artifact_path(model_path) == model_path


def test_resolve_model_path_rejects_missing_path_and_unpromoted_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_model_artifact_path(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="not promoted"):
        resolve_model_artifact_path(tmp_path)


def test_player_stats_loading_errors(tmp_path, monkeypatch):
    missing = tmp_path / "missing.feather"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_player_stats(missing)

    monkeypatch.setattr(pd, "read_feather", Mock(side_effect=RuntimeError("broken")))
    with pytest.raises(ValueError, match="loading player stats"):
        load_player_stats(missing)


@pytest.mark.parametrize(
    ("probabilities", "classes", "message"),
    [
        ([[0.25, 0.75]], [1], None),
        ([0.25, 0.75], [1], "one binary row"),
        ([[0.25]], [1], "one binary row"),
        ([[0.25, 0.75]], [0, 1], "one class"),
        ([[0.25, 0.75]], 1, "one class"),
        ([[False, True]], [1], "not boolean"),
        ([[0.25, 0.75]], [True], "not boolean"),
        ([[0.25, float("nan")]], [1], "finite"),
        ([[0.25, 1.25]], [1], "finite"),
        ([[0.25, 0.75]], [2], "binary"),
    ],
)
def test_prediction_output_normalization(probabilities, classes, message):
    if message is None:
        assert normalize_prediction_outputs(probabilities, classes) == (0.75, 1)
    else:
        with pytest.raises(ValueError, match=message):
            normalize_prediction_outputs(probabilities, classes)


def test_prediction_output_conversion_errors_are_domain_errors():
    class BadFloat:
        def __float__(self):
            raise TypeError("no conversion")

    with pytest.raises(ValueError, match="malformed"):
        normalize_prediction_outputs(np.array([[0.2, BadFloat()]], dtype=object), [1])


def test_snapshot_validation_defenses():
    incomplete = pd.DataFrame([{"player": "A"}])
    with pytest.raises(ValueError, match="missing columns"):
        resolve_player_snapshots(incomplete, "A", "B")

    stats = pd.DataFrame(
        [
            {"player": "A", "rank": 1, "elo": 2000, "as_of_date": "2025-01-01"},
            {"player": "B", "rank": 2, "elo": 1900, "as_of_date": "2025-01-01"},
        ]
    )
    with pytest.raises(UnknownPlayerSnapshotsError) as captured:
        resolve_player_snapshots(stats, "A", "C")
    assert captured.value.players == ["C"]

    duplicate = pd.DataFrame(
        [
            {"player": "A", "rank": 1, "elo": 2000, "as_of_date": "2025-01-01"},
            {"player": "B", "rank": 2, "elo": 1900, "as_of_date": "2025-01-01"},
            {"player": "A", "rank": 1, "elo": 2000, "as_of_date": "2025-01-01"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        resolve_player_snapshots(duplicate, "A", "B")


def test_feature_freshness_allows_current_snapshots():
    snapshot = PlayerSnapshot("A", 1.0, 2000.0, date(2025, 1, 1))
    freshness = calculate_feature_freshness(date(2025, 1, 1), snapshot, snapshot)
    assert freshness.p1_age_days == freshness.p2_age_days == 0


def test_model_bundle_type_validation(monkeypatch, tmp_path):
    path = tmp_path / "model.joblib"
    monkeypatch.setattr(predict_winner, "resolve_model_artifact_path", lambda _: path)
    monkeypatch.setattr(
        predict_winner,
        "read_json",
        lambda _: {
            "model": {"kind": "elo_baseline", "filename": path.name, "sha256": "0" * 64}
        },
    )
    monkeypatch.setattr(predict_winner, "load", lambda _: object())
    with pytest.raises(ValueError, match="elo_baseline"):
        load_model_bundle(path)

    monkeypatch.setattr(
        predict_winner,
        "read_json",
        lambda _: {
            "model": {"kind": "unsupported", "filename": path.name, "sha256": "0" * 64}
        },
    )
    with pytest.raises(ValueError, match="Unsupported"):
        load_model_bundle(path)


def test_successful_metadata_validation_without_artifact_path():
    selection = validate_model_metadata(canonical_metadata())
    assert selection["status"] == "selected"


def test_selection_requires_each_candidate_metrics_section():
    candidates = canonical_metadata()["candidates"]
    del candidates["elo"]["metrics"]
    with pytest.raises(ValueError, match="elo.metrics"):
        compute_selection(candidates)


def test_load_player_stats_success(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "player": "A",
                "rank": 1,
                "elo": 2000,
                "as_of_date": "2025-01-01",
            }
        ]
    )
    monkeypatch.setattr(pd, "read_feather", lambda _: frame.copy())
    loaded = load_player_stats("stats.feather")
    assert loaded.index.tolist() == ["A"]
    assert loaded.loc["A", "as_of_date"] == pd.Timestamp("2025-01-01")


def test_snapshot_rejects_missing_date():
    stats = pd.DataFrame(
        [
            {"player": "A", "rank": 1, "elo": 2000, "as_of_date": None},
            {"player": "B", "rank": 2, "elo": 1900, "as_of_date": "2025-01-01"},
        ]
    )
    with pytest.raises(ValueError, match="snapshot date is missing"):
        resolve_player_snapshots(stats, "A", "B")


def test_lightgbm_bundle_requires_pipeline_and_classifier(monkeypatch, tmp_path):
    path = tmp_path / "model.joblib"
    metadata = {
        "model": {
            "kind": "lightgbm",
            "filename": path.name,
            "sha256": "0" * 64,
        }
    }
    monkeypatch.setattr(predict_winner, "resolve_model_artifact_path", lambda _: path)
    monkeypatch.setattr(predict_winner, "read_json", lambda _: metadata)
    monkeypatch.setattr(predict_winner, "load", lambda _: object())
    with pytest.raises(ValueError, match="metadata kind 'lightgbm'"):
        load_model_bundle(path)

    from acebet.features import FeatureContractPipeline

    monkeypatch.setattr(
        predict_winner, "load", lambda _: FeatureContractPipeline(steps=[])
    )
    with pytest.raises(ValueError, match="missing its fitted LGBMClassifier"):
        load_model_bundle(path)


def test_valid_lightgbm_bundle_reaches_feature_contract_validation(
    monkeypatch, tmp_path
):
    from lightgbm import LGBMClassifier

    from acebet.features import FeatureContractPipeline

    path = tmp_path / "model.joblib"
    metadata = {
        "model": {
            "kind": "lightgbm",
            "filename": path.name,
            "sha256": "0" * 64,
        }
    }
    model = FeatureContractPipeline(steps=[("gbm", LGBMClassifier())])
    compatible = Mock()
    monkeypatch.setattr(predict_winner, "resolve_model_artifact_path", lambda _: path)
    monkeypatch.setattr(predict_winner, "read_json", lambda _: metadata)
    monkeypatch.setattr(predict_winner, "load", lambda _: model)
    monkeypatch.setattr(predict_winner, "assert_model_artifact_compatible", compatible)

    bundle = load_model_bundle(path)

    assert bundle.estimator is model
    assert bundle.identity.kind == "lightgbm"
    compatible.assert_called_once_with(model)


def test_loaded_player_identity_preserves_stored_whitespace(tmp_path):
    stats = pd.DataFrame(
        {
            "player": [" Alpha ", "Beta"],
            "rank": [1, 2],
            "elo": [2000.0, 1900.0],
            "as_of_date": ["2025-01-01"] * 2,
        }
    )
    path = tmp_path / "stats.feather"
    stats.to_feather(path)
    loaded = load_player_stats(path)
    assert loaded.loc[" Alpha ", "player"] == " Alpha "
