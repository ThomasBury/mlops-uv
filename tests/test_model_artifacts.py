import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

import acebet.app.dependencies.predict_winner as prediction_module
from acebet import __version__
from acebet.app.dependencies.predict_winner import load_model_bundle
from acebet.model_artifacts import (
    model_card_path,
    model_metadata_path,
    promotion_record_path,
    read_json,
    resolve_model_artifact_path,
    sha256_file,
    write_json,
)
from acebet.train.promote import promote_model
from acebet.train.train import EloBaselineClassifier, train_model


def create_model_bundle(model_dir: Path) -> Path:
    result = train_model(
        "2015-03-04",
        "2017-03-04",
        output_dir=model_dir,
        export_joblib=True,
    )
    assert result.exported_model_path is not None
    return result.exported_model_path


def test_promote_model_writes_checksum_bound_v2_record(tmp_path):
    model_path = create_model_bundle(tmp_path)

    record_path = promote_model(model_path=model_path)

    record = read_json(record_path)
    assert record["schema_version"] == 2
    assert record["model_filename"] == model_path.name
    assert record["model_kind"] == "elo_baseline"
    assert record["package_version"] == __version__
    assert record["model_sha256"] == sha256_file(model_path)
    assert record["metadata_sha256"] == sha256_file(model_metadata_path(model_path))
    assert record["modelcard_sha256"] == sha256_file(model_card_path(model_path))
    assert record["selection_reason"] == "elo_preferred_challenger_margin_not_met"


def test_directory_loading_requires_promotion_pointer(tmp_path):
    model_path = create_model_bundle(tmp_path)

    with pytest.raises(FileNotFoundError, match="not promoted"):
        resolve_model_artifact_path(tmp_path)

    assert isinstance(load_model_bundle(model_path).estimator, EloBaselineClassifier)


def test_promoted_directory_loads_selected_elo(tmp_path):
    model_path = create_model_bundle(tmp_path)
    promote_model(model_path=model_path)

    assert resolve_model_artifact_path(tmp_path) == model_path
    assert isinstance(load_model_bundle(tmp_path).estimator, EloBaselineClassifier)


def test_cli_export_and_promotion_load_in_fresh_serving_process(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "acebet.train.train",
            "--export-joblib",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    subprocess.run(
        [sys.executable, "-m", "acebet.train.promote", "--output-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import numpy as np
import pandas as pd
from acebet.app.dependencies.predict_winner import load_model_bundle
from acebet.features import MODEL_FEATURE_COLUMNS

model = load_model_bundle(sys.argv[1]).estimator
assert type(model).__module__ == 'acebet.train.estimators'
assert 'acebet.train.train' not in sys.modules
features = pd.DataFrame(0.0, index=[0, 1], columns=MODEL_FEATURE_COLUMNS)
features['proba_elo'] = [0.2, 0.8]
np.testing.assert_allclose(model.predict_proba(features)[:, 1], features['proba_elo'])
""",
            str(tmp_path),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_loaded_bundle_identity_comes_from_validated_metadata(tmp_path):
    model_path = create_model_bundle(tmp_path)
    promote_model(model_path=model_path)
    metadata = read_json(model_metadata_path(model_path))

    bundle = load_model_bundle(tmp_path)

    assert isinstance(bundle.estimator, EloBaselineClassifier)
    assert bundle.artifact_path == model_path.resolve()
    assert bundle.identity.model_dump() == {
        "kind": metadata["model"]["kind"],
        "version": model_path.stem,
        "sha256": metadata["model"]["sha256"],
    }


def test_schema_v1_metadata_and_pointer_are_rejected_actionably(tmp_path):
    model_path = tmp_path / "model_legacy.joblib"
    model_path.write_bytes(b"not loaded")
    model_card_path(model_path).write_text("# legacy", encoding="utf-8")
    write_json(model_metadata_path(model_path), {"schema_version": 1})

    with pytest.raises(ValueError, match="schema v1.*Regenerate and promote"):
        load_model_bundle(model_path)

    write_json(promotion_record_path(tmp_path), {"schema_version": 1})
    with pytest.raises(ValueError, match="schema v1.*Regenerate and promote"):
        load_model_bundle(tmp_path)


def test_direct_loading_requires_both_v2_sidecars(tmp_path):
    model_path = tmp_path / "model_incomplete.joblib"
    model_path.write_bytes(b"not loaded")

    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_model_bundle(model_path)


def test_promotion_retains_training_version_across_runtime_version_changes(tmp_path):
    model_path = create_model_bundle(tmp_path)
    metadata = read_json(model_metadata_path(model_path))
    metadata["model"]["package"]["version"] = "0.2.2"
    write_json(model_metadata_path(model_path), metadata)
    pointer = promote_model(model_path)
    assert read_json(pointer)["package_version"] == "0.2.2"
    assert isinstance(load_model_bundle(tmp_path).estimator, EloBaselineClassifier)


@pytest.mark.parametrize("target", ["model", "metadata", "modelcard", "pointer"])
def test_tampering_fails_before_joblib_load(tmp_path, monkeypatch, target):
    model_path = create_model_bundle(tmp_path)
    pointer_path = promote_model(model_path=model_path)
    paths = {
        "model": model_path,
        "metadata": model_metadata_path(model_path),
        "modelcard": model_card_path(model_path),
        "pointer": pointer_path,
    }
    path = paths[target]
    if target == "pointer":
        record = json.loads(path.read_text(encoding="utf-8"))
        record["model_sha256"] = "0" * 64
        write_json(path, record)
    else:
        path.write_bytes(path.read_bytes() + b"tampered")
    mocked_load = Mock(side_effect=AssertionError("joblib.load must not run"))
    monkeypatch.setattr(prediction_module, "load", mocked_load)

    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_model_bundle(tmp_path)
    mocked_load.assert_not_called()


def test_recorded_selection_is_recomputed_before_joblib_load(tmp_path, monkeypatch):
    model_path = create_model_bundle(tmp_path)
    metadata = read_json(model_metadata_path(model_path))
    metadata["selection"]["reason_code"] = "tampered_reason"
    write_json(model_metadata_path(model_path), metadata)
    mocked_load = Mock(side_effect=AssertionError("joblib.load must not run"))
    monkeypatch.setattr(prediction_module, "load", mocked_load)

    with pytest.raises(ValueError, match="selection.*does not match"):
        load_model_bundle(model_path)
    mocked_load.assert_not_called()


@pytest.mark.parametrize("failure", ["write", "replace", "cleanup"])
def test_failed_promotion_preserves_previous_pointer_and_loadable_bundle(
    tmp_path, monkeypatch, failure
):
    import shutil
    import acebet.model_artifacts as artifacts

    packaged = Path(prediction_module.__file__).resolve().parents[2] / "data"
    for source in packaged.glob("model_2026-08-02-12-59-07.*"):
        shutil.copyfile(source, tmp_path / source.name)
    pointer = tmp_path / "promoted_model.json"
    shutil.copyfile(packaged / pointer.name, pointer)
    previous_bytes = pointer.read_bytes()
    expected_error = OSError("simulated promotion failure")
    temporary_paths = []

    if failure == "write":

        def partial_write(_payload, stream, **_kwargs):
            temporary_paths.append(Path(stream.name))
            stream.write('{"partial":')
            stream.flush()
            raise expected_error

        monkeypatch.setattr(artifacts.json, "dump", partial_write)
    else:

        def fail_replace(source, _destination):
            temporary_paths.append(source)
            raise expected_error

        monkeypatch.setattr(Path, "replace", fail_replace)
        if failure == "cleanup":

            def fail_cleanup(_path, **_kwargs):
                raise PermissionError("simulated cleanup failure")

            monkeypatch.setattr(Path, "unlink", fail_cleanup)

    with pytest.raises(OSError) as error:
        promote_model(output_dir=tmp_path)
    assert error.value is expected_error
    assert pointer.read_bytes() == previous_bytes
    assert isinstance(load_model_bundle(tmp_path).estimator, EloBaselineClassifier)
    assert len(temporary_paths) == 1
    if failure != "cleanup":
        assert not temporary_paths[0].exists()
    else:
        monkeypatch.undo()
        temporary_paths[0].unlink()


@pytest.mark.parametrize("existing_mode", [None, 0o640, 0o644])
def test_atomic_json_preserves_format_and_read_permissions(tmp_path, existing_mode):
    import stat

    destination = tmp_path / "metadata.json"
    if existing_mode is not None:
        destination.write_text("old", encoding="utf-8")
        destination.chmod(existing_mode)
    assert write_json(destination, {"z": 1, "a": [2]}) == destination
    assert (
        destination.read_text(encoding="utf-8")
        == json.dumps({"z": 1, "a": [2]}, indent=2, sort_keys=True) + "\n"
    )
    assert stat.S_IMODE(destination.stat().st_mode) == (existing_mode or 0o644)
    assert list(tmp_path.iterdir()) == [destination]


def test_concurrent_exports_with_same_timestamp_keep_both_artifacts(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timezone
    from threading import Barrier
    from joblib import load
    from acebet.train.train import export_model

    barrier = Barrier(2)
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def export(value):
        barrier.wait(timeout=5)
        return value, export_model({"value": value}, tmp_path, exported_at=timestamp)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(export, (1, 2)))
    assert len(list(tmp_path.iterdir())) == 2
    for value, path in results:
        assert load(path) == {"value": value}
