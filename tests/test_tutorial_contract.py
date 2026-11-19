"""Executable contracts between tutorial claims and the committed implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from html import unescape

import markdown

from fastapi.testclient import TestClient
import pytest

from acebet.app.dependencies.auth import get_demo_users_db
from acebet.app.main import create_app
from acebet.settings import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = PROJECT_ROOT / "docs" / "tutorial"
PACKAGE_DATA_DIR = PROJECT_ROOT / "src" / "acebet" / "data"
MODEL_STEM = "model_2026-08-02-12-59-07"
REQUEST_PAYLOAD = {
    "p1_name": "Fognini F.",
    "p2_name": "Jarry N.",
    "date": "2018-03-04",
    "series": "ATP250",
    "court": "Outdoor",
    "surface": "Hard",
    "round": "1st Round",
    "best_of": 3,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_json_example(source: str) -> dict[str, object]:
    match = re.search(r"```json\n(?P<payload>.*?)\n```", source, flags=re.DOTALL)
    assert match is not None
    return json.loads(match.group("payload"))


def _canonical_documented_response() -> dict[str, object]:
    reference = (PROJECT_ROOT / "docs" / "api" / "sync-endpoints.md").read_text(
        encoding="utf-8"
    )
    section = reference.split("## Canonical Packaged-Bundle Response", maxsplit=1)[1]
    return _first_json_example(section)


@pytest.mark.api
def test_documented_canonical_response_matches_real_packaged_bundle(monkeypatch):
    monkeypatch.setenv("ACEBET_SECRET_KEY", "phase7-contract-secret")
    monkeypatch.setenv("ACEBET_ENV", "test")
    monkeypatch.setenv("ACEBET_RATE_LIMITING_ENABLED", "false")
    monkeypatch.setenv("ACEBET_DEMO_USERNAME", "johndoe")
    monkeypatch.setenv("ACEBET_DEMO_PASSWORD", "secret")
    get_settings.cache_clear()
    get_demo_users_db.cache_clear()
    try:
        with TestClient(create_app()) as client:
            token_response = client.post(
                "/token", data={"username": "johndoe", "password": "secret"}
            )
            assert token_response.status_code == 200
            token = token_response.json()["access_token"]
            response = client.post(
                "/predict/production",
                headers={"Authorization": f"Bearer {token}"},
                json=REQUEST_PAYLOAD,
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload == _canonical_documented_response()
        introduction = (TUTORIAL_DIR / "index.md").read_text(encoding="utf-8")
        assert _first_json_example(introduction) == {
            "p1_win_probability": payload["p1_win_probability"],
            "model": {"version": payload["model"]["version"]},
            "feature_freshness": {
                "p1_as_of_date": payload["feature_freshness"]["p1_as_of_date"],
            },
        }
    finally:
        get_demo_users_db.cache_clear()
        get_settings.cache_clear()


def test_documented_selection_and_bundle_match_committed_records():
    model_path = PACKAGE_DATA_DIR / f"{MODEL_STEM}.joblib"
    metadata_path = PACKAGE_DATA_DIR / f"{MODEL_STEM}.metadata.json"
    modelcard_path = PACKAGE_DATA_DIR / f"{MODEL_STEM}.modelcard.md"
    promotion_path = PACKAGE_DATA_DIR / "promoted_model.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    model_chapter = (
        TUTORIAL_DIR / "02-feature-contract-training-and-export.md"
    ).read_text(encoding="utf-8")
    artifact_guide = (
        PROJECT_ROOT / "docs/user-guide/model-artifacts-and-promotion.md"
    ).read_text(encoding="utf-8")

    assert metadata["dataset"]["row_counts"] == {
        "total": 5236,
        "train": 3482,
        "test": 1754,
    }
    rows = _rendered_table_rows(model_chapter)
    for name, label in (
        ("constant", "Constant"),
        ("elo", "Elo"),
        ("lightgbm", "LightGBM"),
    ):
        expected = [label] + [
            f"{metadata['candidates'][name]['metrics'][metric]:.4f}"
            for metric in ("accuracy", "log_loss", "brier_score", "roc_auc")
        ]
        assert expected in rows
        assert [
            label,
            f"{metadata['candidates'][name]['metrics']['log_loss']:.4f}",
        ] in _rendered_table_rows(artifact_guide)

    selection = metadata["selection"]
    assert selection["selected_model_kind"] == "elo_baseline"
    assert selection["reason_code"] == "elo_preferred_challenger_margin_not_met"
    assert selection["selected_model_kind"] in model_chapter
    assert selection["reason_code"] in model_chapter

    assert _sha256(model_path) == metadata["model"]["sha256"]
    assert _sha256(model_path) == promotion["model_sha256"]
    assert _sha256(metadata_path) == promotion["metadata_sha256"]
    assert _sha256(modelcard_path) == promotion["modelcard_sha256"]
    assert promotion["selection_reason"] == selection["reason_code"]
    assert promotion["model_kind"] == selection["selected_model_kind"]

    documented_values = {
        promotion["model_filename"],
        promotion["metadata_filename"],
        promotion["modelcard_filename"],
        promotion["model_sha256"],
        promotion["metadata_sha256"],
        promotion["modelcard_sha256"],
        metadata["dataset"]["source_sha256"],
        promotion["model_kind"],
        promotion["selection_reason"],
    }
    for value in documented_values:
        assert value in artifact_guide


def test_notebook_is_clean_non_exporting_and_uses_shared_contracts():
    notebook_path = (
        PROJECT_ROOT
        / "docs"
        / "assets"
        / "notebooks"
        / "01-starter-feature-and-training-flow.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert all(cell.get("execution_count") is None for cell in notebook["cells"])
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    for expected in (
        "prepare_data",
        "PreparedTrainingData",
        "date_grouped_temporal_holdout",
        "MODEL_FEATURE_COLUMNS",
        "evaluate_probabilities",
        "compute_selection",
    ):
        assert expected in source
    for obsolete in ("time_series_split", "just feature-state\n", "just train\n"):
        assert obsolete not in source
    assert "export_model" not in source
    assert "promote_model" not in source


def test_documented_commands_and_docker_claims_match_implementation():
    documentation_paths = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "getting-started.md",
        PROJECT_ROOT / "docs" / "tutorial-shortcuts-vs-production.md",
        PROJECT_ROOT / "docs" / "uv" / "uv-workflows-for-mlops.md",
        PROJECT_ROOT / "docs" / "cicd" / "github-actions-pipeline.md",
        PROJECT_ROOT / "docs" / "deployment" / "docker-and-github-actions.md",
        PROJECT_ROOT / "docs" / "api" / "sync-endpoints.md",
        *sorted(TUTORIAL_DIR.glob("*.md")),
    ]
    documentation = "\n".join(
        path.read_text(encoding="utf-8") for path in documentation_paths
    )

    assert "--frozen" not in documentation
    assert "--no-cache" not in documentation
    assert "`just feature-state`" not in documentation
    assert "`just train`" not in documentation
    assert not re.search(r"\buv sync(?! --locked)", documentation)
    assert "falls back to the latest" not in documentation.lower()
    assert "newest preferred" not in documentation.lower()
    assert not re.search(
        r"just data-build[^\n]*(?:data/atp_data_production\.feather|src/acebet/data)",
        documentation,
    )

    ci_reference = (
        PROJECT_ROOT / "docs" / "cicd" / "github-actions-pipeline.md"
    ).read_text(encoding="utf-8")
    assert "uv sync --locked --dev" in ci_reference
    assert "just docker-smoke" in ci_reference

    deployment = (
        PROJECT_ROOT / "docs" / "deployment" / "docker-and-github-actions.md"
    ).read_text(encoding="utf-8")
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    for claim in (
        "uv sync --locked --no-dev --no-install-project --no-editable",
        "uv sync --locked --no-dev --no-editable --reinstall-package acebet",
        "--mount=type=cache,target=/root/.cache/uv",
        "python:3.12.13-slim-bookworm",
        "copies only `/app/.venv`",
        "appuser",
        "fastapi run --entrypoint acebet.app.main:app --host 0.0.0.0 --port 8000",
        "immutable fallback",
        "ACEBET_MODEL_DIR",
        "ACEBET_DATA_DIR",
        "just docker-build",
        "just docker-smoke",
    ):
        assert claim in deployment
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


def test_ci_permissions_action_pins_and_dependabot_contract():
    main_workflow = (PROJECT_ROOT / ".github/workflows/main.yaml").read_text(
        encoding="utf-8"
    )
    docs_workflow = (PROJECT_ROOT / ".github/workflows/docs.yaml").read_text(
        encoding="utf-8"
    )
    dependabot = (PROJECT_ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")

    action_reference = re.compile(r"(?:- )?uses: [^@\s]+@[0-9a-f]{40}(?:\s+#\s+v\S+)?$")
    for workflow in (main_workflow, docs_workflow):
        uses_lines = [line.strip() for line in workflow.splitlines() if "uses:" in line]
        assert uses_lines
        assert all(action_reference.fullmatch(line) for line in uses_lines)

    global_permissions = docs_workflow.split("jobs:", maxsplit=1)[0]
    build_job, deploy_job = docs_workflow.split("  deploy:", maxsplit=1)
    integration = main_workflow.split("  integration:", maxsplit=1)[1].split(
        "  docker:", maxsplit=1
    )[0]
    assert (
        integration.index("Run integration, API, tracing, and packaging tests")
        < integration.index("Download unit coverage")
        < integration.index("Combine and enforce coverage guarantees")
    )

    assert "contents: read" in global_permissions
    assert "pages: write" not in global_permissions
    assert "id-token: write" not in global_permissions
    assert "pages: write" not in build_job
    assert "pages: write" in deploy_job
    assert "id-token: write" in deploy_job
    assert "name: github-pages" in deploy_job
    assert "group: pages-${{ github.ref }}" in docs_workflow

    assert "package-ecosystem: github-actions" in dependabot
    assert "package-ecosystem: docker" in dependabot
    assert "package-ecosystem: uv" in dependabot
    assert dependabot.count("interval: weekly") == 3


def test_notebook_dependencies_are_development_only():
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dev_group = pyproject.split("[dependency-groups]", maxsplit=1)[1].split(
        "[tool.setuptools]", maxsplit=1
    )[0]
    project_dependencies = pyproject.split("dependencies = [", maxsplit=1)[1].split(
        "]", maxsplit=1
    )[0]
    assert '"ipykernel' in dev_group
    assert '"nbclient' in dev_group
    assert '"ipykernel' not in project_dependencies
    assert '"nbclient' not in project_dependencies


def _rendered_table_rows(source):
    rendered = markdown.markdown(source, extensions=["tables"])
    return [
        [
            unescape(re.sub(r"<[^>]+>", "", cell)).strip()
            for cell in re.findall(r"<t[dh](?:\s[^>]*)?>(.*?)</t[dh]>", row, re.DOTALL)
        ]
        for row in re.findall(r"<tr>(.*?)</tr>", rendered, re.DOTALL)
    ]


def test_rendered_api_table_contains_the_complete_route_surface():
    source = (PROJECT_ROOT / "docs/api/sync-endpoints.md").read_text(encoding="utf-8")
    rows = _rendered_table_rows(source)
    actual = {(row[0], row[1]) for row in rows[1:]}
    assert actual == {
        ("GET", "/"),
        ("GET", "/healthz"),
        ("GET", "/readyz"),
        ("GET", "/metrics"),
        ("POST", "/token"),
        ("GET", "/users/me"),
        ("POST", "/predict/production"),
        ("POST", "/predict/testing"),
    }


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
