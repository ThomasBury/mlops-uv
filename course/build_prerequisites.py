"""Build the M5 and M9a prerequisite archives from the pinned 2021 snapshot."""

import argparse
from pathlib import Path
from shutil import copyfile, copytree
import tarfile
from tempfile import TemporaryDirectory

from mlflow.models import Model

from acebet.dataprep.historical_results import build_results, read_results
from acebet.historical_monitoring import review_candidate
from acebet.model_artifacts import sha256_file, write_json
from acebet.train.historical_evaluation import evaluate_candidate, train_candidate
from m9_bundle import BUNDLE_URI, DATABASES, rebase_uris
from m9_registry import promote, register, registry, review, stage
from m9_rehearsal import delayed_run, synthetic_scores

SNAPSHOT = Path(__file__).resolve().parents[1] / "data/raw/valuebetennis-2021-atp.csv"


def archive_metadata(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def archive_directory(source: Path, destination: Path) -> None:
    with tarfile.open(destination, "x:gz") as archive:
        for path in sorted(source.iterdir()):
            archive.add(path, arcname=path.name, filter=archive_metadata)


def make_registry_portable(lab: Path, original_uri: str) -> None:
    for name in DATABASES:
        rebase_uris(lab / name, original_uri, BUNDLE_URI, portable=True)
    for name in ("archive", "incumbent"):
        for metadata in (lab / name / "mlartifacts/models").glob("*/artifacts/MLmodel"):
            model = Model.load(str(metadata))
            # The logged model ID resolves through the tracking database.
            model.artifact_path = None
            model.save(str(metadata))


def build(output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output = output.resolve()
    with TemporaryDirectory(dir=output.parent) as temporary:
        work = Path(temporary)
        packed = work / "bundles"
        packed.mkdir()
        lab = work / "m9a"
        archive = lab / "archive"
        archive.mkdir(parents=True)
        history = archive / "matches.parquet"
        frozen = archive / "bundle"
        write_json(archive / "preparation.json", build_results(SNAPSHOT, history))
        results = read_results(history)
        train_candidate(results, frozen)
        evaluate_candidate(results, frozen)
        write_json(archive / "validation.json", review_candidate(history, frozen))

        serving = work / "m5"
        serving.mkdir()
        for name in ("matches.parquet", "preparation.json"):
            copyfile(archive / name, serving / name)
        copytree(frozen, serving / "bundle")
        archive_directory(serving, packed / "m5-prerequisites.tar.gz")

        incumbent = lab / "incumbent"
        delayed_run(history, incumbent)
        rehearsal = work / "incumbent-rehearsal"
        synthetic_scores(incumbent, rehearsal)
        client = registry(lab / "registry.db")
        dev = register(client, incumbent)
        approval = review(client, dev["version"], rehearsal)
        production = promote(client, approval, rehearsal)
        write_json(lab / "incumbent-review.json", approval)
        stage(
            client,
            production["production"]["version"],
            incumbent / "matches.parquet",
            lab / "before",
        )
        make_registry_portable(lab, lab.as_uri())
        archive_directory(lab, packed / "m9a-prerequisites.tar.gz")

        checksums = "".join(
            f"{sha256_file(path)}  {path.name}\n"
            for path in sorted(packed.glob("*.tar.gz"))
        )
        (packed / "SHA256SUMS").write_text(checksums, encoding="utf-8")
        packed.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory in an existing parent")
    build(parser.parse_args().output)
