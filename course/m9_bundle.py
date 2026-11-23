"""Open a fresh copy of the local registry and tracked runs shipped with M9a."""

import argparse
from contextlib import closing
from pathlib import Path
import sqlite3
import tarfile
from tempfile import TemporaryDirectory

BUNDLE_URI = "file:///__acebet_m9_bundle__"
DATABASES = ("archive/mlflow.db", "incumbent/mlflow.db", "registry.db")
# The URI columns in the SQLite databases produced by the locked MLflow version.
URI_COLUMNS = (
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("model_versions", "source"),
    ("model_versions", "storage_location"),
)


def rebase_uris(database: Path, old: str, new: str, *, portable: bool = False) -> None:
    with closing(sqlite3.connect(database.as_uri() + "?mode=rw", uri=True)) as db:
        with db:
            for table, column in URI_COLUMNS:
                db.execute(
                    f"UPDATE {table} SET {column} = ? || substr({column}, ?) "
                    f"WHERE {column} = ? OR substr({column}, 1, ?) = ?",
                    (new, len(old) + 1, old, len(old) + 1, old + "/"),
                )
            if portable:
                db.execute(
                    "UPDATE experiments SET artifact_location = ? WHERE experiment_id = 0",
                    (f"{new}/mlruns/0",),
                )
                db.execute("UPDATE runs SET user_id = 'course-fixture'")
                for table, key, value in (
                    ("tags", "key", "value"),
                    ("logged_model_tags", "tag_key", "tag_value"),
                ):
                    db.execute(
                        f"UPDATE {table} SET {value} = 'course-fixture' WHERE {key} = 'mlflow.user'"
                    )
                    db.execute(
                        f"UPDATE {table} SET {value} = 'course/build_prerequisites.py' "
                        f"WHERE {key} = 'mlflow.source.name'"
                    )
                    db.execute(
                        f"DELETE FROM {table} WHERE {key} = 'mlflow.source.git.repoURL'"
                    )
        if portable:
            # Rewrite database pages to drop superseded paths and account names.
            db.execute("VACUUM")


def open_bundle(archive: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    destination = output.resolve()
    with TemporaryDirectory(dir=destination.parent) as temporary:
        root = Path(temporary) / "registry"
        root.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(root, filter="data")
        for name in DATABASES:
            rebase_uris(root / name, BUNDLE_URI, destination.as_uri())
        root.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path, help="New directory in an existing parent")
    args = parser.parse_args()
    open_bundle(args.archive, args.output)


if __name__ == "__main__":
    main()
