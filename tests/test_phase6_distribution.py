"""Distribution contents and isolated installation smoke tests."""

from __future__ import annotations

from email.parser import BytesParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest

pytestmark = pytest.mark.packaging

EXPECTED_DATA_ASSETS = {
    "atp_data_sample.feather",
    "latest_player_stats.feather",
    "latest_player_stats_sample.feather",
    "model_2026-08-02-12-59-07.joblib",
    "model_2026-08-02-12-59-07.metadata.json",
    "model_2026-08-02-12-59-07.modelcard.md",
    "promoted_model.json",
}
EXPECTED_CORE_REQUIREMENTS = {
    "bcrypt",
    "fastapi",
    "fastapi-cli",
    "httpx",
    "python-multipart",
    "uvicorn",
    "joblib",
    "lightgbm",
    "numpy",
    "pandas",
    "pyarrow",
    "pydantic",
    "pydantic-settings",
    "prometheus-client",
    "python-jose",
    "scikit-learn",
    "slowapi",
    "structlog",
}
EXPECTED_TRACING_REQUIREMENTS = {
    "opentelemetry-api",
    "opentelemetry-exporter-otlp-proto-http",
    "opentelemetry-instrumentation-fastapi",
    "opentelemetry-sdk",
}


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    project_root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp("distributions")
    subprocess.run(
        ["uv", "build", "--out-dir", str(output)],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return next(output.glob("acebet-*.whl")), next(output.glob("acebet-*.tar.gz"))


def test_wheel_and_sdist_contain_exact_runtime_assets(distributions):
    wheel, sdist = distributions

    with ZipFile(wheel) as archive:
        wheel_assets = {
            Path(name).name
            for name in archive.namelist()
            if name.startswith("acebet/data/") and not name.endswith("__init__.py")
        }
    assert wheel_assets == EXPECTED_DATA_ASSETS

    with tarfile.open(sdist, "r:gz") as archive:
        sdist_assets = {
            Path(name).name
            for name in archive.getnames()
            if "/src/acebet/data/" in name and not name.endswith("__init__.py")
        }
    assert sdist_assets == EXPECTED_DATA_ASSETS


def test_wheel_metadata_exposes_only_intended_runtime_and_tracing_dependencies(
    distributions,
):
    wheel, _ = distributions
    with ZipFile(wheel) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_name))

    requirements = [
        Requirement(value) for value in metadata.get_all("Requires-Dist", [])
    ]
    core = {
        canonicalize_name(requirement.name)
        for requirement in requirements
        if requirement.marker is None
    }
    tracing = {
        canonicalize_name(requirement.name)
        for requirement in requirements
        if requirement.marker is not None and "tracing" in str(requirement.marker)
    }

    assert core == EXPECTED_CORE_REQUIREMENTS
    by_name = {
        canonicalize_name(item.name): item
        for item in requirements
        if item.marker is None
    }
    assert not by_name["fastapi"].extras
    assert str(by_name["fastapi"].specifier) == "<0.163,>=0.162.5"
    assert str(by_name["fastapi-cli"].specifier) == "<0.1,>=0.0.33"
    assert by_name["uvicorn"].extras == {"standard"}
    assert tracing == EXPECTED_TRACING_REQUIREMENTS
    assert metadata.get_all("Provides-Extra") == ["tracing"]
    assert (
        not (
            {"ipykernel", "matplotlib", "mlflow", "nbclient", "skops"}
            | EXPECTED_TRACING_REQUIREMENTS
        )
        & core
    )


def _venv_python(venv: Path) -> Path:
    executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return venv / executable


def _install_in_clean_venv(venv: Path, requirement: str) -> Path:
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    python = _venv_python(venv)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            requirement,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return python


@pytest.fixture
def clean_venv(tmp_path):
    with TemporaryDirectory(dir=tmp_path) as directory:
        yield Path(directory)


DEFAULT_SMOKE = """
import subprocess
import sys
from importlib.metadata import distributions
installed = {
    item.metadata['Name'].lower().replace('_', '-') for item in distributions()
}
for package in ('ipykernel', 'matplotlib', 'mlflow', 'nbclient', 'skops'):
    assert package not in installed, package
assert 'opentelemetry-api' in installed
assert not any(name == 'opentelemetry-sdk' or name.startswith((
    'opentelemetry-exporter-', 'opentelemetry-instrumentation'
)) for name in installed), installed
from fastapi.testclient import TestClient
from acebet.app.main import app
with TestClient(app) as client:
    assert client.get('/healthz').json() == {'status': 'ok'}
    assert client.get('/readyz').json() == {'status': 'ready'}
    assert type(app.state.runtime_assets.production.model).__name__ == 'EloBaselineClassifier'
training = subprocess.run(
    [sys.executable, '-m', 'acebet.train.train'], capture_output=True, text=True
)
assert training.returncode != 0
assert 'Prepared dataset not found' in training.stderr and '--data-path' in training.stderr
"""

TRACING_SMOKE = """
from importlib.util import find_spec
assert find_spec('opentelemetry.sdk') is not None
assert find_spec('opentelemetry.instrumentation.fastapi') is not None
from fastapi.testclient import TestClient
from acebet.app.main import app
with TestClient(app) as client:
    assert client.get('/readyz').json() == {'status': 'ready'}
assert app.state.tracer_provider is None
"""


@pytest.mark.parametrize("artifact_index", [0, 1], ids=["wheel", "sdist"])
def test_default_clean_install_replays_and_serves_without_optional_packages(
    distributions, tmp_path, artifact_index, clean_venv
):
    artifact = distributions[artifact_index]
    python = _install_in_clean_venv(clean_venv, str(artifact))
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ACEBET_", "PYTHON"))
    }
    replay_command = python.with_name(
        "acebet-replay.exe" if os.name == "nt" else "acebet-replay"
    )
    forward = ["--canonical-probability", "0.70"]
    for args, first, second, probability in (
        ([], "P01", "P02", 0.50),
        (forward, "P01", "P02", 0.70),
        ([*forward, "--reverse"], "P02", "P01", 0.30),
    ):
        replay = subprocess.run(
            [str(replay_command), *args],
            cwd=tmp_path,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads(replay.stdout) == {
            "fixture_id": "F01",
            "cutoff": "2030-06-10T12:00:00Z",
            "p1_id": first,
            "p2_id": second,
            "p1_win_probability": pytest.approx(probability),
        }
    environment.update(
        {
            "ACEBET_SECRET_KEY": "distribution-default-smoke-secret",
            "ACEBET_ENV": "production",
            "ACEBET_DEMO_PASSWORD": "distribution-password",
            "ACEBET_RATE_LIMITING_ENABLED": "false",
        }
    )
    subprocess.run(
        [str(python), "-c", DEFAULT_SMOKE],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_tracing_extra_clean_install_starts_and_shuts_down(
    distributions, tmp_path, clean_venv
):
    wheel, _ = distributions
    python = _install_in_clean_venv(clean_venv, f"{wheel}[tracing]")
    environment = os.environ.copy()
    environment.update(
        {
            "ACEBET_SECRET_KEY": "distribution-tracing-smoke-secret",
            "ACEBET_ENV": "production",
            "ACEBET_DEMO_PASSWORD": "distribution-password",
            "ACEBET_RATE_LIMITING_ENABLED": "false",
            "ACEBET_TRACING_ENABLED": "true",
            "ACEBET_OTLP_ENDPOINT": "http://127.0.0.1:4318",
        }
    )
    subprocess.run(
        [str(python), "-c", TRACING_SMOKE],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
