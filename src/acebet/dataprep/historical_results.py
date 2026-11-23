"""Build the tutorial's canonical results from a pinned Valuebetennis extract."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import csv
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_URL = "https://www.valuebetennis.com/en/donnees.htm"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
SNAPSHOT_SHA256 = "ca16daa1c89b085680152dc41924cad7e2791775c23d68a083c2c0237cdf0598"
SOURCE_COLUMNS = (
    "match_id",
    "date",
    "tournoi",
    "tournoi_id",
    "categorie",
    "genre",
    "surface",
    "tour",
    "joueur1",
    "joueur1_id",
    "joueur2",
    "joueur2_id",
    "cote1_ouverture",
    "cote2_ouverture",
    "cote1_cloture",
    "cote2_cloture",
    "vainqueur_id",
    "score",
    "duree_min",
)
SURFACES = {
    "dur": "Hard",
    "dur intérieur": "Hard",
    "terre battue": "Clay",
    "gazon": "Grass",
}
ID_PATTERN = re.compile(r"[1-9][0-9]*")
SET_PATTERN = re.compile(r"([0-9]{1,2})-([0-9]{1,2})(\([0-9]+\))?")
SCHEMA_VERSION = b"1"
RESULT_SCHEMA = pa.schema(
    [
        pa.field("match_id", pa.string(), nullable=False),
        pa.field("event_date", pa.date32(), nullable=False),
        pa.field("available_at", pa.timestamp("ms", tz="UTC"), nullable=False),
        pa.field("surface", pa.string(), nullable=False),
        pa.field("p1_id", pa.string(), nullable=False),
        pa.field("p2_id", pa.string(), nullable=False),
        pa.field("p1_name", pa.string(), nullable=False),
        pa.field("p2_name", pa.string(), nullable=False),
        pa.field("label", pa.int8(), nullable=False),
    ]
)


@dataclass(frozen=True)
class MatchResult:
    """A result in player-ID order, with a separate availability timestamp."""

    match_id: str
    event_date: date
    available_at: datetime
    surface: str
    p1_id: str
    p2_id: str
    p1_name: str
    p2_name: str
    label: int

    def __post_init__(self) -> None:
        if (
            not all(
                value.strip()
                for value in (
                    self.match_id,
                    self.p1_id,
                    self.p2_id,
                    self.p1_name,
                    self.p2_name,
                )
            )
            or self.p1_id >= self.p2_id
        ):
            raise ValueError(
                f"Missing identity or invalid player order for {self.match_id}."
            )
        if self.label not in (0, 1) or self.surface not in SURFACES.values():
            raise ValueError(f"Invalid label or surface for {self.match_id}.")
        if (
            self.available_at.utcoffset() != timedelta(0)
            or self.available_at.date() <= self.event_date
        ):
            raise ValueError(
                f"Availability must follow the event day in UTC for {self.match_id}."
            )


def ingest_snapshot(path: Path) -> bytes:
    """Read and verify the exact snapshot bytes that will be normalized."""
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != SNAPSHOT_SHA256:
        raise ValueError(f"Unexpected raw snapshot SHA-256 for {path}.")
    return raw


def _completed_winner_slot(score: str) -> int | None:
    """Return the score's winning source slot for an unambiguous best-of-three."""
    # ponytail: only ordinary best-of-three sets, extend when accepting other formats.
    winners = []
    for token in score.split():
        match = SET_PATTERN.fullmatch(token)
        if match is None:
            return None
        first, second = int(match[1]), int(match[2])
        high, low = max(first, second), min(first, second)
        if not (
            (high == 6 and low <= 4 and match[3] is None)
            or (high == 7 and low == 5 and match[3] is None)
            or (high == 7 and low == 6)
        ):
            return None
        winners.append(int(second > first))
    if len(winners) not in (2, 3):
        return None
    if len(winners) == 3 and winners[0] == winners[1]:
        return None
    winner = winners[-1]
    return winner if winners.count(winner) == 2 else None


def _normalize_row(row: dict[str, str]) -> MatchResult | None:
    match_id = row["match_id"]
    player_ids = (row["joueur1_id"], row["joueur2_id"])
    names = (row["joueur1"], row["joueur2"])
    winner_id = row["vainqueur_id"]
    if (
        not all(ID_PATTERN.fullmatch(value) for value in (match_id, *player_ids))
        or player_ids[0] == player_ids[1]
        or not all(name.strip() for name in names)
        or (winner_id and winner_id not in player_ids)
    ):
        raise ValueError(f"Invalid match, player, or winner identity for {match_id!r}.")
    event_date = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S").date()
    if (
        event_date.year != 2021
        or row["genre"] != "atp"
        or row["categorie"] not in {"Main tour", "Masters"}
        or not 4 <= int(row["tour"]) <= 12
        or row["surface"] not in SURFACES
    ):
        raise ValueError(f"Out-of-scope date, tour, round, or surface for {match_id}.")
    winner_slot = _completed_winner_slot(row["score"])
    if winner_slot is None or player_ids[winner_slot] != winner_id:
        return None
    (p1_id, p1_name), (p2_id, p2_name) = sorted(zip(player_ids, names))
    # ponytail: the export gives no publication times, replace this delay with observed timestamps when available.
    available_at = datetime.combine(
        event_date + timedelta(days=2), time.min, timezone.utc
    )
    return MatchResult(
        match_id,
        event_date,
        available_at,
        SURFACES[row["surface"]],
        p1_id,
        p2_id,
        p1_name,
        p2_name,
        int(winner_id == p1_id),
    )


def _ordered(results: Iterable[MatchResult]) -> list[MatchResult]:
    return sorted(results, key=lambda row: (row.event_date, row.match_id))


def normalize_results(raw: bytes) -> list[MatchResult]:
    """Validate source rows, deduplicate repeats, and keep completed matches."""
    results, _ = _normalize_with_counts(raw)
    return results


def _normalize_with_counts(raw: bytes) -> tuple[list[MatchResult], dict[str, int]]:
    reader = csv.DictReader(
        StringIO(raw.decode("utf-8-sig")), delimiter=";", strict=True
    )
    if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
        raise ValueError("Unexpected Valuebetennis CSV columns.")
    seen = {}
    results = []
    duplicate_rows = 0
    for row in reader:
        if None in row or None in row.values():
            raise ValueError(f"Incorrect column count on CSV line {reader.line_num}.")
        match_id = row["match_id"]
        if match_id in seen:
            if seen[match_id] != row:
                raise ValueError(f"Conflicting duplicate for match {match_id}.")
            duplicate_rows += 1
            continue
        seen[match_id] = row
        result = _normalize_row(row)
        if result is not None:
            results.append(result)
    if not results:
        raise ValueError("No completed matches in the snapshot.")
    return _ordered(results), {
        "input_rows": len(seen) + duplicate_rows,
        "duplicate_rows": duplicate_rows,
        "excluded_rows": len(seen) - len(results),
    }


def content_sha256(results: Iterable[MatchResult]) -> str:
    """Hash sorted logical content independently of the Parquet writer."""
    digest = sha256(SCHEMA_VERSION + b"\n")
    for row in _ordered(results):
        record = asdict(row)
        record["event_date"] = row.event_date.isoformat()
        record["available_at"] = row.available_at.isoformat()
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def build_results(snapshot: Path, output: Path) -> dict[str, str | int]:
    """Publish canonical Parquet and report its identity and row counts."""
    if snapshot.resolve() == output.resolve():
        raise ValueError("The Parquet output must preserve the raw snapshot.")
    results, counts = _normalize_with_counts(ingest_snapshot(snapshot))
    digest = content_sha256(results)
    schema = RESULT_SCHEMA.with_metadata(
        {
            b"schema_version": SCHEMA_VERSION,
            b"source_sha256": SNAPSHOT_SHA256.encode(),
            b"content_sha256": digest.encode(),
            b"source": SOURCE_URL.encode(),
            b"license": LICENSE_URL.encode(),
            b"attribution": b"Valuebetennis, Tennis results and odds since 2021",
            b"changes": b"ATP subset; completed matches; UTC days; canonical player order",
        }
    )
    table = pa.Table.from_pylist([asdict(row) for row in results], schema=schema)
    with NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        pq.write_table(table, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "source_sha256": SNAPSHOT_SHA256,
        "content_sha256": digest,
        "rows": len(results),
        **counts,
    }


def read_results(path: Path) -> list[MatchResult]:
    """Load canonical results only when schema, source, and content agree."""
    table = pq.ParquetFile(path).read()
    metadata = table.schema.metadata or {}
    if (
        not table.schema.equals(RESULT_SCHEMA)
        or metadata.get(b"schema_version") != SCHEMA_VERSION
        or metadata.get(b"source_sha256") != SNAPSHOT_SHA256.encode()
        or any(column.null_count for column in table.columns)
    ):
        raise ValueError("Unexpected canonical Parquet schema or source.")
    results = [MatchResult(**row) for row in table.to_pylist()]
    if not results or len({row.match_id for row in results}) != len(results):
        raise ValueError("Canonical results require nonempty, unique match IDs.")
    if metadata.get(b"content_sha256") != content_sha256(results).encode():
        raise ValueError("Canonical content SHA-256 does not match.")
    return _ordered(results)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build_results(args.snapshot, args.output)))


if __name__ == "__main__":
    main()
