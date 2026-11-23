"""Contracts for the packaged model, application build, CI, and CLI."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DATA_DIR = PROJECT_ROOT / "src" / "acebet" / "data"
MODEL_STEM = "model_2026-08-02-12-59-07"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_packaged_selection_and_bundle_match_committed_records():
    model_path = PACKAGE_DATA_DIR / f"{MODEL_STEM}.joblib"
    metadata_path = PACKAGE_DATA_DIR / f"{MODEL_STEM}.metadata.json"
    modelcard_path = PACKAGE_DATA_DIR / f"{MODEL_STEM}.modelcard.md"
    promotion_path = PACKAGE_DATA_DIR / "promoted_model.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    assert metadata["dataset"]["row_counts"] == {
        "total": 5236,
        "train": 3482,
        "test": 1754,
    }
    selection = metadata["selection"]
    assert selection["selected_model_kind"] == "elo_baseline"
    assert selection["reason_code"] == "elo_preferred_challenger_margin_not_met"

    assert _sha256(model_path) == metadata["model"]["sha256"]
    assert _sha256(model_path) == promotion["model_sha256"]
    assert _sha256(metadata_path) == promotion["metadata_sha256"]
    assert _sha256(modelcard_path) == promotion["modelcard_sha256"]
    assert promotion["selection_reason"] == selection["reason_code"]
    assert promotion["model_kind"] == selection["selected_model_kind"]

    assert (
        _sha256(PACKAGE_DATA_DIR / "latest_player_stats.feather")
        == metadata["player_stats_sha256"]
    )
    post = metadata["post_selection"]
    assert (
        post["date_range"]["start_date"]
        > metadata["dataset"]["observed_window"]["end_date"]
    )
    card = modelcard_path.read_text(encoding="utf-8")
    assert f"{post['rows']:,}" in card
    assert f"{post['metrics']['elo']['log_loss']:.5f}" in card


def test_docker_build_installs_only_the_application_environment():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "uv sync --locked --no-dev --no-editable --reinstall-package acebet"
        in dockerfile
    )
    assert dockerfile.count("uv sync --locked --no-dev") == 2
    assert dockerfile.count("FROM ${PYTHON_IMAGE}") == 2
    assert "COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv" in (
        dockerfile
    )
    assert "COPY --from=builder --chown=appuser:appuser /app /app" not in dockerfile


def test_ci_action_pins_and_dependabot_contract():
    main_workflow = (PROJECT_ROOT / ".github/workflows/main.yaml").read_text(
        encoding="utf-8"
    )
    dependabot = (PROJECT_ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")

    action_reference = re.compile(r"(?:- )?uses: [^@\s]+@[0-9a-f]{40}(?:\s+#\s+v\S+)?$")
    uses_lines = [
        line.strip() for line in main_workflow.splitlines() if "uses:" in line
    ]
    assert uses_lines
    assert all(action_reference.fullmatch(line) for line in uses_lines)

    assert "package-ecosystem: github-actions" in dependabot
    assert "package-ecosystem: docker" in dependabot
    assert "package-ecosystem: uv" in dependabot
    assert dependabot.count("interval: weekly") == 3


@pytest.mark.parametrize(
    "module",
    [
        "acebet.train.train",
        "acebet.train.promote",
        "acebet.dataprep.dataprep",
        "acebet.dataprep.raw_to_prepared",
    ],
)
def test_cli_help_explains_arguments_and_displays_defaults(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    help_text = " ".join(result.stdout.split())
    assert (
        "--input-path" in help_text
        if ".dataprep." in module
        else "--output-dir" in help_text
    )
    if module != "acebet.dataprep.raw_to_prepared":
        assert "(default:" in help_text
    if module == "acebet.train.train":
        assert (
            "Start of the training/selection window (inclusive). (default: 2015-03-04)"
            in help_text
        )
        assert (
            "End of the training/selection window (inclusive). (default: 2017-03-04)"
            in help_text
        )
