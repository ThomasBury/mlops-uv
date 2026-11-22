"""Starter for the parcel delivery transfer exercise."""

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from math import log, log1p
from pathlib import Path


@dataclass(frozen=True)
class Booking:
    parcel_id: str
    depot: str
    booked_at: datetime
    promised_at: datetime
    delivered_at: datetime | None
    outcome_available_at: datetime | None


@dataclass(frozen=True)
class StatusEvent:
    parcel_id: str
    depot: str
    occurred_at: datetime
    available_at: datetime
    status: str


def load_data(directory: Path) -> tuple[list[Booking], list[StatusEvent]]:
    """Read the two exercise CSVs and parse their UTC timestamps."""
    with (directory / "bookings.csv").open(newline="", encoding="utf-8") as stream:
        bookings = [
            Booking(
                parcel_id=row["parcel_id"],
                depot=row["depot"],
                booked_at=datetime.fromisoformat(row["booked_at"]),
                promised_at=datetime.fromisoformat(row["promised_at"]),
                delivered_at=datetime.fromisoformat(row["delivered_at"])
                if row["delivered_at"]
                else None,
                outcome_available_at=datetime.fromisoformat(row["outcome_available_at"])
                if row["outcome_available_at"]
                else None,
            )
            for row in csv.DictReader(stream)
        ]
    with (directory / "status_events.csv").open(newline="", encoding="utf-8") as stream:
        events = [
            StatusEvent(
                parcel_id=row["parcel_id"],
                depot=row["depot"],
                occurred_at=datetime.fromisoformat(row["occurred_at"]),
                available_at=datetime.fromisoformat(row["available_at"]),
                status=row["status"],
            )
            for row in csv.DictReader(stream)
        ]
    return bookings, events


def midnight_utc(day: date) -> datetime:
    return datetime.combine(day, time.min, tzinfo=timezone.utc)


def label_at(booking: Booking, as_of: datetime) -> int | None:
    """Return the lateness label if it is known at as_of."""
    raise NotImplementedError("Implement the outcome contract.")


def booking_feature(booking: Booking, events: list[StatusEvent]) -> int:
    """Return 1 for a capacity warning on the latest eligible depot event, else 0."""
    history = (
        event
        for event in events
        if event.depot == booking.depot
        and event.parcel_id != booking.parcel_id
        and event.occurred_at <= booking.booked_at
    )
    latest = max(history, key=lambda event: event.occurred_at, default=None)
    return int(latest is not None and latest.status == "capacity_warning")


def log_loss(label: int, probability: float) -> float:
    """Score one parcel. The exercise uses probabilities strictly inside (0, 1)."""
    return -log(probability) if label == 1 else -log1p(-probability)


def evaluate(
    bookings: list[Booking],
    events: list[StatusEvent],
    train_at: datetime,
    score_at: datetime,
) -> dict:
    """Return training IDs, paired predictions, and their losses."""
    raise NotImplementedError("Implement the chronological comparison.")


def run(
    evaluator: Callable[[list[Booking], list[StatusEvent], datetime, datetime], dict],
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("train_date", type=date.fromisoformat)
    parser.add_argument("score_date", type=date.fromisoformat)
    args = parser.parse_args()
    if args.train_date >= args.score_date:
        parser.error("train_date must be earlier than score_date")
    bookings, events = load_data(args.data)
    report = evaluator(
        bookings, events, midnight_utc(args.train_date), midnight_utc(args.score_date)
    )
    report["train_date"] = args.train_date.isoformat()
    report["score_date"] = args.score_date.isoformat()
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    run(evaluate)
