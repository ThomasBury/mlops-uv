"""Review exact MLflow versions and prepare local historical forecast releases."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from acebet.historical_forecast import load_forecaster
from acebet.historical_monitoring import review_candidate
from acebet.model_artifacts import read_json, sha256_file, write_json

DEV = "acebet-dev"
PROD = "acebet-prod"
ALIAS = "champion"
ASSETS = ("selection.json", "model.joblib")


def registry(db: Path) -> MlflowClient:
    uri = f"sqlite:///{db.resolve()}"
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def identity(version: ModelVersion) -> dict:
    return {"model": version.name, "version": version.version, "run_id": version.run_id}


def register(client: MlflowClient, run: Path) -> dict:
    selection = read_json(run / "bundle/selection.json")
    tracking_db = run / "mlflow.db"
    if not tracking_db.is_file():
        raise ValueError("Use the original tracked run directory to register a model.")
    tracked = registry(tracking_db).get_run(selection["run_id"])
    model_hash = selection["artifacts_sha256"]["model.joblib"]
    if (
        tracked.info.status != "FINISHED"
        or tracked.data.params.get("model_sha256") != model_hash
    ):
        raise ValueError("The finished tracking run must identify the frozen model.")

    try:
        client.create_registered_model(DEV)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    return identity(
        client.create_model_version(DEV, tracked.info.artifact_uri, tracked.info.run_id)
    )


def version_assets(
    client: MlflowClient, version: ModelVersion, destination: Path
) -> Path:
    # Use the version's files instead of reserializing the fitted pipeline.
    source = client.get_model_version_download_uri(version.name, version.version)
    return Path(download_artifacts(artifact_uri=source, dst_path=str(destination)))


def review(client: MlflowClient, version: str, run: Path) -> dict:
    bundle = run / "bundle"
    dev = client.get_model_version(DEV, version)
    selection = read_json(bundle / "selection.json")
    with TemporaryDirectory() as temporary:
        assets = version_assets(client, dev, Path(temporary))
        if dev.run_id != selection["run_id"] or any(
            sha256_file(assets / name) != sha256_file(bundle / name) for name in ASSETS
        ):
            raise ValueError(
                "The development version does not match the reviewed bundle."
            )

    return {
        "development": identity(dev),
        "rehearsal": read_json(bundle / "final/evaluation.json").get("rehearsal"),
        **review_candidate(run / "matches.parquet", bundle),
    }


def promote(client: MlflowClient, approval: dict, run: Path) -> dict:
    dev = approval["development"]
    if approval != review(client, dev["version"], run):
        raise ValueError("The review no longer agrees with the version and evidence.")
    if not approval["gate"]["accepted"]:
        raise ValueError("The Elo gate retained the incumbent.")

    source = f"models:/{DEV}/{dev['version']}"
    review_hash = sha256(json.dumps(approval, sort_keys=True).encode()).hexdigest()
    tags = {
        "selection_sha256": approval["selection_sha256"],
        "evaluation_sha256": approval["evaluation_sha256"],
        "model_sha256": approval["fixture"]["model_sha256"],
        "history_sha256": approval["fixture"]["history_sha256"],
        "rehearsal": json.dumps(approval["rehearsal"], sort_keys=True),
        "review_sha256": review_hash,
    }
    # ponytail: one local operator. Use coordinated approvals for concurrent writers.
    previous_copy = next(
        (
            v
            for v in client.search_model_versions(f"name = '{PROD}'")
            if v.source == source and v.tags.get("review_sha256") == review_hash
        ),
        None,
    )
    prod = previous_copy or client.copy_model_version(source, PROD)
    # Write the review tag last so a retry only reuses a fully recorded version.
    for key, value in tags.items():
        client.set_model_version_tag(PROD, prod.version, key, value)
    client.set_registered_model_alias(PROD, ALIAS, prod.version)
    return {"development": dev, "production": identity(prod), "source": source, **tags}


def select_champion(client: MlflowClient, version: str) -> dict:
    prod = client.get_model_version(PROD, version)
    if "review_sha256" not in prod.tags:
        raise ValueError("Select a production version with recorded approval.")
    client.set_registered_model_alias(PROD, ALIAS, prod.version)
    return identity(prod)


def stage(client: MlflowClient, version: str, history: Path, output: Path) -> dict:
    prod = client.get_model_version(PROD, version)
    if "review_sha256" not in prod.tags:
        raise ValueError("Stage a production version with recorded approval.")
    if output.exists():
        raise FileExistsError(output)

    with TemporaryDirectory(dir=output.parent) as temporary:
        work = Path(temporary)
        assets = version_assets(client, prod, work)
        release = work / "release"
        release.mkdir()
        for name in ASSETS:
            copyfile(assets / name, release / name)
        copyfile(history, release / "matches.parquet")
        frozen = load_forecaster(release / "matches.parquet", release)
        if (
            sha256_file(release / "selection.json") != prod.tags["selection_sha256"]
            or frozen.model_sha256 != prod.tags["model_sha256"]
            or frozen.history_sha256 != prod.tags["history_sha256"]
        ):
            raise ValueError(
                "Release assets differ from the approved production version."
            )
        for name in (*ASSETS, "matches.parquet"):
            (release / name).chmod(0o644)
        manifest = {"production": identity(prod), "source": prod.source, **prod.tags}
        write_json(release / "release.json", manifest)
        release.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("register").add_argument("run", type=Path)
    check = commands.add_parser("review")
    check.add_argument("version")
    check.add_argument("run", type=Path)
    approve = commands.add_parser("promote")
    approve.add_argument("review", type=Path)
    approve.add_argument("run", type=Path)
    prepare = commands.add_parser("stage")
    prepare.add_argument("version")
    prepare.add_argument("history", type=Path)
    prepare.add_argument("output", type=Path)
    commands.add_parser("champion").add_argument("version", nargs="?")
    args = parser.parse_args()
    client = registry(args.db)

    try:
        match args.command:
            case "register":
                result = register(client, args.run)
            case "review":
                result = review(client, args.version, args.run)
            case "promote":
                result = promote(client, read_json(args.review), args.run)
            case "stage":
                result = stage(client, args.version, args.history, args.output)
            case "champion" if args.version is not None:
                result = select_champion(client, args.version)
            case "champion":
                result = identity(client.get_model_version_by_alias(PROD, ALIAS))
    except (ValueError, OSError, MlflowException) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
