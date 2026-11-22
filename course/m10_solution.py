"""Worked implementation for M10. Open after attempting the exercise."""

from datetime import datetime
from statistics import mean

from m10_starter import Booking, StatusEvent, log_loss, run


def label_at(booking: Booking, as_of: datetime) -> int | None:
    if (
        booking.delivered_at is None
        or booking.outcome_available_at is None
        or booking.outcome_available_at > as_of
    ):
        return None
    return int(booking.delivered_at > booking.promised_at)


def booking_feature(booking: Booking, events: list[StatusEvent]) -> int:
    history = (
        event
        for event in events
        if event.depot == booking.depot
        and event.parcel_id != booking.parcel_id
        and event.available_at <= booking.booked_at
    )
    latest = max(history, key=lambda event: event.occurred_at, default=None)
    return int(latest is not None and latest.status == "capacity_warning")


def evaluate(
    bookings: list[Booking],
    events: list[StatusEvent],
    train_at: datetime,
    score_at: datetime,
) -> dict:
    training_ids = []
    labels_by_feature = {0: [], 1: []}
    for booking in bookings:
        if (
            booking.booked_at < train_at
            and (label := label_at(booking, train_at)) is not None
        ):
            training_ids.append(booking.parcel_id)
            labels_by_feature[booking_feature(booking, events)].append(label)

    training_labels = labels_by_feature[0] + labels_by_feature[1]
    if set(training_labels) != {0, 1}:
        raise ValueError("Choose a training date with both eligible label classes.")
    baseline = mean(training_labels)
    candidate = {
        feature: (sum(labels) + 1) / (len(labels) + 2)
        for feature, labels in labels_by_feature.items()
    }

    predictions = []
    baseline_losses, candidate_losses = [], []
    for booking in bookings:
        if (
            train_at <= booking.booked_at <= score_at
            and (label := label_at(booking, score_at)) is not None
        ):
            probability = candidate[booking_feature(booking, events)]
            predictions.append(
                {
                    "parcel_id": booking.parcel_id,
                    "label": label,
                    "baseline_probability": baseline,
                    "candidate_probability": probability,
                }
            )
            baseline_losses.append(log_loss(label, baseline))
            candidate_losses.append(log_loss(label, probability))

    if not predictions:
        raise ValueError(
            "Choose a scoring date with at least one eligible evaluation label."
        )
    return {
        "training_ids": training_ids,
        "predictions": predictions,
        "mean_log_loss": {
            "baseline": mean(baseline_losses),
            "candidate": mean(candidate_losses),
        },
        "mean_paired_difference": mean(
            candidate_loss - baseline_loss
            for baseline_loss, candidate_loss in zip(
                baseline_losses, candidate_losses, strict=True
            )
        ),
    }


if __name__ == "__main__":
    run(evaluate)
