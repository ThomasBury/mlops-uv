"""Check a learner's M10 feature, outcome contract, and saved comparison."""

import argparse
import json
import runpy
from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
from math import isclose, log
from pathlib import Path
from statistics import mean

from m10_starter import load_data, midnight_utc

DATA = Path(__file__).with_name("fixtures") / "m10"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_features(feature, bookings, events, expected_features):
    booking = bookings["P08"]
    late_warning = next(event for event in events if event.parcel_id == "P07")
    eligible_status = next(event for event in events if event.parcel_id == "P05")

    def replace_event(original, **changes):
        return [
            replace(event, **changes) if event == original else event
            for event in events
        ]

    original = feature(booking, events)
    changed_late = feature(booking, replace_event(late_warning, status="on_track"))
    require(
        original == changed_late,
        "P08 changed when a status received after booking was edited.",
    )
    changed_earlier = feature(
        booking, replace_event(eligible_status, status="capacity_warning")
    )
    require(
        original == 0 and changed_earlier == 1,
        "P08 must respond to a change in eligible depot history.",
    )
    at_cutoff = feature(
        booking, replace_event(late_warning, available_at=booking.booked_at)
    )
    require(at_cutoff == 1, "A status available at the booking cutoff is eligible.")

    for parcel_id, expected in expected_features.items():
        require(
            feature(bookings[parcel_id], events) == expected,
            f"Incorrect booking-time feature for {parcel_id}.",
        )


def check_labels(label_at, bookings):
    on_time = bookings["P01"]
    late = bookings["P02"]
    unresolved = bookings["P12"]
    before = on_time.outcome_available_at - timedelta(seconds=1)
    require(
        label_at(on_time, before) is None,
        "P01 was scored before its outcome was available.",
    )
    require(
        label_at(on_time, on_time.outcome_available_at) == 0,
        "P01 was delivered at the promise and is on time.",
    )
    require(
        label_at(late, late.outcome_available_at) == 1,
        "P02 was delivered after the promise and is late.",
    )
    require(
        label_at(unresolved, unresolved.booked_at + timedelta(days=30)) is None,
        "P12 has no observed outcome.",
    )


def check_report(report, bookings, features):
    train_at = midnight_utc(date.fromisoformat(report["train_date"]))
    score_at = midnight_utc(date.fromisoformat(report["score_date"]))
    require(train_at < score_at, "Training must precede scoring.")
    outcomes = {
        parcel_id: int(booking.delivered_at > booking.promised_at)
        for parcel_id, booking in bookings.items()
        if booking.delivered_at is not None and booking.outcome_available_at is not None
    }
    training = {
        parcel_id: label
        for parcel_id, label in outcomes.items()
        if bookings[parcel_id].booked_at < train_at
        and bookings[parcel_id].outcome_available_at <= train_at
    }
    scoring = {
        parcel_id: label
        for parcel_id, label in outcomes.items()
        if train_at <= bookings[parcel_id].booked_at <= score_at
        and bookings[parcel_id].outcome_available_at <= score_at
    }
    require(set(training.values()) == {0, 1}, "Training must include both labels.")
    require(bool(scoring), "There are no eligible evaluation parcels.")
    require(
        Counter(report["training_ids"]) == Counter(training.keys()),
        f"Training IDs must match those with labels available by the training boundary. Expected {sorted(training)}.",
    )
    predictions = report["predictions"]
    require(
        Counter(row["parcel_id"] for row in predictions) == Counter(scoring.keys()),
        f"Expected one paired prediction per eligible evaluation parcel. Expected {sorted(scoring)}.",
    )

    baseline = mean(training.values())
    groups = Counter(
        (features[parcel_id], label) for parcel_id, label in training.items()
    )
    baseline_losses, candidate_losses = [], []
    for row in predictions:
        parcel_id = row["parcel_id"]
        label = scoring[parcel_id]
        require(row["label"] == label, f"Incorrect label for {parcel_id}.")
        feature = features[parcel_id]
        n_late, n_on_time = groups[feature, 1], groups[feature, 0]
        candidate = (n_late + 1) / (n_late + n_on_time + 2)
        for model, expected, losses in (
            ("baseline", baseline, baseline_losses),
            ("candidate", candidate, candidate_losses),
        ):
            probability = row[f"{model}_probability"]
            require(
                0 < probability < 1,
                f"{parcel_id}: {model} probability must be in (0, 1).",
            )
            require(
                isclose(probability, expected, abs_tol=1e-12, rel_tol=0),
                f"{parcel_id}: wrong {model} probability.",
            )
            losses.append(-log(probability if label == 1 else 1 - probability))

    for model, losses in (
        ("baseline", baseline_losses),
        ("candidate", candidate_losses),
    ):
        require(
            isclose(
                report["mean_log_loss"][model], mean(losses), abs_tol=1e-12, rel_tol=0
            ),
            f"Reported {model} loss does not match the submitted predictions.",
        )
    paired_difference = mean(
        c - b for b, c in zip(baseline_losses, candidate_losses, strict=True)
    )
    require(
        isclose(
            report["mean_paired_difference"],
            paired_difference,
            abs_tol=1e-12,
            rel_tol=0,
        ),
        "Reported paired difference does not match the submitted predictions.",
    )
    print(
        f"PASS: {len(training)} training parcels, {len(scoring)} paired predictions and reproducible loss"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("learner", type=Path)
    parser.add_argument(
        "report", type=Path, nargs="?", help="Omit to check the feature alone."
    )
    args = parser.parse_args()
    rows, events = load_data(DATA)
    bookings = {booking.parcel_id: booking for booking in rows}

    # Walk newest first and take the first event known at booking.
    newest_first = sorted(events, key=lambda event: event.occurred_at, reverse=True)
    expected_features = {}
    for booking in rows:
        for event in newest_first:
            if (
                event.available_at <= booking.booked_at
                and event.depot == booking.depot
                and event.parcel_id != booking.parcel_id
            ):
                expected_features[booking.parcel_id] = int(
                    event.status == "capacity_warning"
                )
                break
        else:
            expected_features[booking.parcel_id] = 0

    try:
        learner = runpy.run_path(str(args.learner))
        check_features(learner["booking_feature"], bookings, events, expected_features)
        print("PASS: booking-time feature and availability mutations")
        if args.report is not None:
            check_labels(learner["label_at"], bookings)
            print(
                "PASS: on-time, late and unresolved outcomes, including delayed availability"
            )
            report = json.loads(args.report.read_text(encoding="utf-8"))
            check_report(report, bookings, expected_features)
    except (KeyError, NotImplementedError, OSError, TypeError, ValueError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
