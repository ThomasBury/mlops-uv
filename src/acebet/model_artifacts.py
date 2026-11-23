"""Schema-v2 model bundle integrity, selection, and promotion helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
import stat
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import IO, Any, cast

MODEL_METADATA_SCHEMA_VERSION = 2
PROMOTION_RECORD_SCHEMA_VERSION = 2
PROMOTION_RECORD_FILENAME = "promoted_model.json"
PACKAGE_NAME = "acebet"
MODEL_SERIALIZATION = "joblib"
LEARNED_VS_CONSTANT_LOG_LOSS_MARGIN = 0.005
LIGHTGBM_VS_ELO_LOG_LOSS_MARGIN = 0.005
METRIC_NAMES = ("accuracy", "log_loss", "brier_score", "roc_auc")
MODEL_KINDS = frozenset({"elo_baseline", "lightgbm"})


@dataclass(frozen=True)
class ValidatedModelArtifact:
    """Metadata and model bytes from the same validated bundle."""

    path: Path
    metadata: dict[str, Any]
    model_bytes: bytes
    metadata_sha256: str


def model_metadata_path(model_path: str | Path) -> Path:
    """Return the JSON metadata sidecar path for an exported model."""
    return Path(model_path).with_suffix(".metadata.json")


def model_card_path(model_path: str | Path) -> Path:
    """Return the Markdown model-card path for an exported model."""
    return Path(model_path).with_suffix(".modelcard.md")


def promotion_record_path(model_dir: str | Path) -> Path:
    """Return the promotion record path for a model directory."""
    return Path(model_dir) / PROMOTION_RECORD_FILENAME


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Replace JSON atomically after a complete write beside the destination."""

    def write(stream: IO[str]) -> None:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")

    return write_atomic(path, write)


def write_atomic(path: str | Path, write: Callable[[IO[str]], object]) -> Path:
    """Replace a text file after writing it beside the destination."""
    output_path = Path(path)
    mode = stat.S_IMODE(output_path.stat().st_mode) if output_path.exists() else 0o644
    stream = NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(stream.name)
    try:
        with stream:
            write(stream.file)
        os.chmod(temporary_path, mode)
        temporary_path.replace(output_path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
    return output_path


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    path = Path(path)
    return _decode_json_object(path.read_text(encoding="utf-8"), path)


def _decode_json_object(text: str, path: Path) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in '{path}'.")
    return payload


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest for the complete file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_schema(payload: Mapping[str, Any], expected: int, kind: str) -> None:
    observed = payload.get("schema_version")
    if observed != expected:
        if observed == 1:
            raise ValueError(
                f"{kind} schema v1 is no longer supported. Regenerate and promote the bundle using the current training workflow."
            )
        raise ValueError(
            f"Unsupported {kind} schema version {observed!r}; expected {expected}. Regenerate and promote the bundle using the current training workflow."
        )


def _candidate_metrics(candidates: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    entry = candidates.get(name)
    if not isinstance(entry, Mapping) or not isinstance(entry.get("metrics"), Mapping):
        raise ValueError(f"Candidate metadata is missing '{name}.metrics'.")
    return entry["metrics"]


def _margin_met(delta: float, margin: float) -> bool:
    """Treat exact decimal policy boundaries as passing despite binary float noise."""
    return delta > margin or math.isclose(delta, margin, rel_tol=0.0, abs_tol=1e-12)


def compute_selection(candidates: Mapping[str, Any]) -> dict[str, Any]:
    """Select the servable candidate using unrounded log-loss comparisons."""
    losses: dict[str, float] = {}
    for name in ("constant", "elo", "lightgbm"):
        raw = _candidate_metrics(candidates, name).get("log_loss")
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ValueError(f"Candidate '{name}' log_loss must be numeric.")
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"Candidate '{name}' log_loss must be finite and non-negative."
            )
        losses[name] = value

    deltas = {
        "constant_minus_elo_log_loss": losses["constant"] - losses["elo"],
        "constant_minus_lightgbm_log_loss": losses["constant"] - losses["lightgbm"],
        "elo_minus_lightgbm_log_loss": losses["elo"] - losses["lightgbm"],
    }
    elo_servable = _margin_met(
        deltas["constant_minus_elo_log_loss"], LEARNED_VS_CONSTANT_LOG_LOSS_MARGIN
    )
    lightgbm_servable = _margin_met(
        deltas["constant_minus_lightgbm_log_loss"],
        LEARNED_VS_CONSTANT_LOG_LOSS_MARGIN,
    )
    if not elo_servable and not lightgbm_servable:
        status, selected, code = "failed", None, "no_candidate_beats_constant"
        reason = "Neither non-constant candidate beats constant log loss by the required 0.005 margin."
    elif elo_servable and not lightgbm_servable:
        status, selected, code = "selected", "elo_baseline", "elo_only_servable"
        reason = (
            "Elo is the only non-constant candidate that beats the constant baseline."
        )
    elif lightgbm_servable and not elo_servable:
        status, selected, code = "selected", "lightgbm", "lightgbm_only_servable"
        reason = "LightGBM is the only non-constant candidate that beats the constant baseline."
    elif _margin_met(
        deltas["elo_minus_lightgbm_log_loss"], LIGHTGBM_VS_ELO_LOG_LOSS_MARGIN
    ):
        status, selected, code = "selected", "lightgbm", "lightgbm_beats_elo_margin"
        reason = "Both candidates are servable and LightGBM beats Elo by the required 0.005 margin."
    else:
        status, selected, code = (
            "selected",
            "elo_baseline",
            "elo_preferred_challenger_margin_not_met",
        )
        reason = "Both candidates are servable, but LightGBM does not beat Elo by the required 0.005 margin."
    return {
        "primary_metric": "log_loss",
        "learned_vs_constant_margin": LEARNED_VS_CONSTANT_LOG_LOSS_MARGIN,
        "lightgbm_vs_elo_margin": LIGHTGBM_VS_ELO_LOG_LOSS_MARGIN,
        "status": status,
        "selected_model_kind": selected,
        "deltas": deltas,
        "reason_code": code,
        "reason": reason,
    }


def candidates_with_selection_details(
    candidate_metrics: Mapping[str, Mapping[str, float]],
    training_prevalence: float,
    lightgbm_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Build candidate metadata with derived improvements and servability flags."""
    candidates: dict[str, Any] = {
        "training_prevalence": float(training_prevalence),
        "constant": {"metrics": dict(candidate_metrics["constant"])},
        "elo": {"metrics": dict(candidate_metrics["elo"])},
        "lightgbm": {
            "metrics": dict(candidate_metrics["lightgbm"]),
            "parameters": dict(lightgbm_parameters),
        },
    }
    selection = compute_selection(candidates)
    for name, delta_key in (
        ("elo", "constant_minus_elo_log_loss"),
        ("lightgbm", "constant_minus_lightgbm_log_loss"),
    ):
        improvement = selection["deltas"][delta_key]
        candidates[name]["log_loss_improvement_over_constant"] = improvement
        candidates[name]["servable"] = _margin_met(
            improvement, LEARNED_VS_CONSTANT_LOG_LOSS_MARGIN
        )
    return candidates


def _validate_metrics(candidates: Mapping[str, Any]) -> None:
    for name in ("constant", "elo", "lightgbm"):
        metrics = _candidate_metrics(candidates, name)
        if set(metrics) != set(METRIC_NAMES):
            raise ValueError(
                f"Candidate '{name}' metrics must contain exactly {list(METRIC_NAMES)}."
            )
        for metric_name, raw in metrics.items():
            if (
                not isinstance(raw, (int, float))
                or isinstance(raw, bool)
                or not math.isfinite(float(raw))
            ):
                raise ValueError(
                    f"Candidate '{name}' metric '{metric_name}' must be finite and numeric."
                )
        if not 0 <= float(metrics["accuracy"]) <= 1 or float(metrics["log_loss"]) < 0:
            raise ValueError(f"Candidate '{name}' accuracy or log_loss is invalid.")
        if (
            not 0 <= float(metrics["brier_score"]) <= 1
            or not 0 <= float(metrics["roc_auc"]) <= 1
        ):
            raise ValueError(f"Candidate '{name}' brier_score or roc_auc is invalid.")


def validate_model_metadata(
    metadata: Mapping[str, Any], *, model_path: str | Path | None = None
) -> dict[str, Any]:
    """Validate v2 package identity, feature contract, selection, and model digest."""
    _require_schema(metadata, MODEL_METADATA_SCHEMA_VERSION, "model metadata")
    required = {"dataset", "feature_contract", "candidates", "selection", "model"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Model metadata schema v2 is missing sections: {missing}.")

    dataset = metadata["dataset"]
    dataset_keys = {
        "source_path",
        "source_sha256",
        "requested_window",
        "observed_window",
        "holdout_cutoff",
        "train_date_range",
        "test_date_range",
        "row_counts",
    }
    if not isinstance(dataset, Mapping) or set(dataset) != dataset_keys:
        raise ValueError(
            "Model metadata dataset section is incomplete or has unknown fields."
        )
    if "player_stats_sha256" in metadata and not _is_sha256(
        metadata["player_stats_sha256"]
    ):
        raise ValueError("Player stats sha256 must be a lowercase SHA-256 digest.")
    source_path = dataset["source_path"]
    if (
        not isinstance(source_path, str)
        or not source_path
        or Path(source_path).is_absolute()
    ):
        raise ValueError(
            "Dataset source_path must be a non-empty project-relative path."
        )
    source_sha = dataset["source_sha256"]
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 64
        or any(character not in "0123456789abcdef" for character in source_sha)
    ):
        raise ValueError("Dataset source_sha256 must be a lowercase SHA-256 digest.")
    row_counts = dataset["row_counts"]
    if (
        not isinstance(row_counts, Mapping)
        or set(row_counts) != {"total", "train", "test"}
        or any(
            not isinstance(row_counts[key], int) or row_counts[key] <= 0
            for key in row_counts
        )
        or row_counts["total"] != row_counts["train"] + row_counts["test"]
    ):
        raise ValueError(
            "Dataset row_counts must contain consistent positive total/train/test counts."
        )
    for key in (
        "requested_window",
        "observed_window",
        "train_date_range",
        "test_date_range",
    ):
        window = dataset[key]
        if not isinstance(window, Mapping) or set(window) != {"start_date", "end_date"}:
            raise ValueError(f"Dataset {key} must contain start_date and end_date.")
    if not isinstance(dataset["holdout_cutoff"], str):
        raise ValueError("Dataset holdout_cutoff must be an ISO date string.")

    from acebet.features import (
        CATEGORICAL_FEATURE_COLUMNS,
        MODEL_FEATURE_COLUMNS,
        NUMERIC_FEATURE_COLUMNS,
    )

    expected_contract = {
        "columns": list(MODEL_FEATURE_COLUMNS),
        "numeric_columns": list(NUMERIC_FEATURE_COLUMNS),
        "categorical_columns": list(CATEGORICAL_FEATURE_COLUMNS),
        "count": len(MODEL_FEATURE_COLUMNS),
    }
    if metadata["feature_contract"] != expected_contract:
        raise ValueError(
            "Model artifact metadata feature contract mismatch with the current ordered contract."
        )
    candidates = metadata["candidates"]
    if not isinstance(candidates, Mapping):
        raise ValueError("Model metadata candidates section must be an object.")
    _validate_metrics(candidates)
    prevalence = candidates.get("training_prevalence")
    if (
        not isinstance(prevalence, (int, float))
        or isinstance(prevalence, bool)
        or not 0 <= float(prevalence) <= 1
    ):
        raise ValueError("Training prevalence must be numeric and in [0, 1].")
    recomputed = compute_selection(candidates)
    for name, delta_key in (
        ("elo", "constant_minus_elo_log_loss"),
        ("lightgbm", "constant_minus_lightgbm_log_loss"),
    ):
        entry = candidates[name]
        expected_improvement = recomputed["deltas"][delta_key]
        if entry.get("log_loss_improvement_over_constant") != expected_improvement:
            raise ValueError(
                f"Candidate '{name}' improvement over constant is inconsistent."
            )
        expected_servable = _margin_met(
            expected_improvement, LEARNED_VS_CONSTANT_LOG_LOSS_MARGIN
        )
        if entry.get("servable") is not expected_servable:
            raise ValueError(f"Candidate '{name}' servability flag is inconsistent.")
    if not isinstance(candidates["lightgbm"].get("parameters"), Mapping):
        raise ValueError("LightGBM candidate parameters are missing.")
    selection = metadata["selection"]
    # The reason is display text and may differ across schema-v2 exports.
    if (
        not isinstance(selection, Mapping)
        or not isinstance(selection.get("reason"), str)
        or {**selection, "reason": recomputed["reason"]} != recomputed
    ):
        raise ValueError(
            "Recorded model selection does not match selection recomputed from candidate metrics."
        )
    if recomputed["status"] != "selected":
        raise ValueError(
            "Model metadata has no servable selected candidate; regenerate the bundle."
        )
    model = metadata["model"]
    if (
        not isinstance(model, Mapping)
        or model.get("kind") not in MODEL_KINDS
        or model.get("kind") != recomputed["selected_model_kind"]
    ):
        raise ValueError("Model kind does not match the recomputed selected candidate.")
    package = model.get("package")
    if (
        not isinstance(package, Mapping)
        or package.get("name") != PACKAGE_NAME
        or not isinstance(package.get("version"), str)
        or not package["version"].strip()
    ):
        raise ValueError("Model artifact package identity is invalid.")
    if (
        model.get("serialization") != MODEL_SERIALIZATION
        or model.get("trusted_only") is not True
    ):
        raise ValueError(
            "Model metadata must identify trusted-only joblib serialization."
        )
    model_sha = model.get("sha256")
    if not _is_sha256(model_sha):
        raise ValueError("Model sha256 must be a lowercase SHA-256 digest.")
    if model_path is not None:
        path = Path(model_path)
        _validate_model_file(model, path, sha256_file(path))
    return recomputed


def _validate_model_file(model: Mapping[str, Any], path: Path, digest: str) -> None:
    if model.get("filename") != path.name:
        raise ValueError("Model metadata filename does not match the model file.")
    if model.get("sha256") != digest:
        raise ValueError(
            "Model artifact SHA-256 mismatch. The bundle may be incomplete or tampered."
        )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def ensure_artifact_bundle(model_path: str | Path) -> tuple[Path, Path, Path]:
    """Ensure the model artifact and both sidecars exist."""
    model_file = Path(model_path)
    metadata_file, card_file = (
        model_metadata_path(model_file),
        model_card_path(model_file),
    )
    missing = [
        str(path)
        for path in (model_file, metadata_file, card_file)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Model artifact bundle is incomplete. Missing: " + ", ".join(missing)
        )
    return model_file, metadata_file, card_file


def validate_artifact_bundle(model_path: str | Path) -> dict[str, Any]:
    """Validate a complete direct-file bundle without deserializing joblib."""
    return _read_artifact_bundle(*ensure_artifact_bundle(model_path)).metadata


def _read_artifact_bundle(
    model_file: Path,
    metadata_file: Path,
    card_file: Path,
    record: Mapping[str, Any] | None = None,
) -> ValidatedModelArtifact:
    model_bytes = model_file.read_bytes()
    model_digest = hashlib.sha256(model_bytes).hexdigest()
    metadata_bytes = metadata_file.read_bytes()
    metadata_digest = hashlib.sha256(metadata_bytes).hexdigest()
    if record is not None:
        for key, digest in (
            ("model_sha256", model_digest),
            ("metadata_sha256", metadata_digest),
            ("modelcard_sha256", sha256_file(card_file)),
        ):
            if record.get(key) != digest:
                raise ValueError(
                    f"Promotion record {key} mismatch. The bundle may be incomplete or tampered."
                )
    metadata = _decode_json_object(metadata_bytes.decode("utf-8"), metadata_file)
    validate_model_metadata(metadata)
    _validate_model_file(metadata["model"], model_file, model_digest)
    return ValidatedModelArtifact(model_file, metadata, model_bytes, metadata_digest)


def list_exported_models(model_dir: str | Path) -> list[Path]:
    """Return timestamped model artifacts for trusted promotion tooling."""
    return sorted(Path(model_dir).glob("model_*.joblib"))


def resolve_latest_model_path(model_dir: str | Path) -> Path:
    """Return the latest exported model for trusted promotion tooling only."""
    models = list_exported_models(model_dir)
    if not models:
        raise FileNotFoundError(
            f"No trained model found in '{model_dir}'. Expected a 'model_*.joblib' file."
        )
    return max(models, key=lambda file: file.name)


def build_promotion_record(model_path: str | Path) -> dict[str, Any]:
    """Validate a v2 bundle and build its checksum-bound promotion record."""
    model_file, metadata_file, card_file = ensure_artifact_bundle(model_path)
    artifact = _read_artifact_bundle(model_file, metadata_file, card_file)
    metadata = artifact.metadata
    selection, model = metadata["selection"], metadata["model"]
    return {
        "schema_version": PROMOTION_RECORD_SCHEMA_VERSION,
        "promoted_at_utc": metadata["exported_at_utc"],
        "model_filename": model_file.name,
        "metadata_filename": metadata_file.name,
        "modelcard_filename": card_file.name,
        "model_kind": model["kind"],
        "package_name": PACKAGE_NAME,
        "package_version": model["package"]["version"],
        "model_sha256": model["sha256"],
        "metadata_sha256": artifact.metadata_sha256,
        "modelcard_sha256": sha256_file(card_file),
        "selection_reason": selection["reason_code"],
        "selection_reason_text": selection["reason"],
    }


def validate_promotion_record(record: Mapping[str, Any], model_dir: str | Path) -> Path:
    """Validate pointer hashes and identities before deserializing joblib."""
    return _read_promoted_bundle(record, Path(model_dir)).path


def _read_promoted_bundle(
    record: Mapping[str, Any], directory: Path
) -> ValidatedModelArtifact:
    _require_schema(record, PROMOTION_RECORD_SCHEMA_VERSION, "promotion record")
    names = {
        key: record.get(f"{key}_filename") for key in ("model", "metadata", "modelcard")
    }
    if any(
        not isinstance(value, str) or Path(value).name != value
        for value in names.values()
    ):
        raise ValueError("Promotion record filenames must be simple local filenames.")
    model_file, metadata_file, card_file = (
        directory / cast(str, names[key]) for key in ("model", "metadata", "modelcard")
    )
    missing = [
        str(path)
        for path in (model_file, metadata_file, card_file)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Promotion record points to missing model artifact files: "
            + ", ".join(missing)
        )
    if (
        metadata_file.name != model_metadata_path(model_file).name
        or card_file.name != model_card_path(model_file).name
    ):
        raise ValueError(
            "Promotion record sidecar filenames do not match the model filename."
        )
    artifact = _read_artifact_bundle(model_file, metadata_file, card_file, record)
    model = artifact.metadata["model"]
    identity = {
        "model_kind": model["kind"],
        "package_name": PACKAGE_NAME,
        "package_version": model["package"]["version"],
        "selection_reason": artifact.metadata["selection"]["reason_code"],
    }
    mismatched = [key for key, value in identity.items() if record.get(key) != value]
    if mismatched:
        raise ValueError(
            "Promotion record identity does not match model metadata: "
            + ", ".join(mismatched)
        )
    return artifact


def resolve_model_artifact_path(model_path: str | Path) -> Path:
    """Resolve and validate a trusted direct file or required promoted directory."""
    return read_validated_model_artifact(model_path).path


def read_validated_model_artifact(model_path: str | Path) -> ValidatedModelArtifact:
    """Return checked model bytes and metadata for a direct file or promoted directory."""
    path = Path(model_path)
    if path.is_file():
        pointer = promotion_record_path(path.parent)
        if pointer.is_file():
            record = read_json(pointer)
            if record.get("model_filename") == path.name:
                return _read_promoted_bundle(record, path.parent)
        return _read_artifact_bundle(*ensure_artifact_bundle(path))
    if not path.is_dir():
        raise FileNotFoundError(f"Model path '{path}' does not exist.")
    pointer = promotion_record_path(path)
    if not pointer.is_file():
        raise FileNotFoundError(
            f"Model directory '{path}' is not promoted: missing '{PROMOTION_RECORD_FILENAME}'. Promote a complete v2 bundle first."
        )
    return _read_promoted_bundle(read_json(pointer), path)
