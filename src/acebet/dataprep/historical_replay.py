"""Replay pre-match Elo from results and corrections visible by each cutoff."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from itertools import groupby
import json
from math import fsum, log
from pathlib import Path

from acebet.dataprep.historical_results import MatchResult, read_results
from acebet.features import elo_win_probability
from acebet.forecast import F01_CUTOFF, F01_ID, F01_PLAYERS

INITIAL_ELO = 1500.0
K = 32.0
DEVELOPMENT_END = date(2021, 9, 1)
F01_SURFACE = "Clay"


@dataclass(frozen=True)
class WinnerCorrection:
    """A new winner observation kept separately from the original result."""

    match_id: str
    winner_id: str
    available_at: datetime

    def __post_init__(self) -> None:
        if self.available_at.utcoffset() != timedelta(0):
            raise ValueError("Correction availability must be in UTC.")


def results_as_of(
    results: Iterable[MatchResult],
    cutoff: datetime,
    corrections: Iterable[WinnerCorrection] = (),
) -> list[MatchResult]:
    """Select the visible version of each result in deterministic event-day order."""
    if cutoff.utcoffset() != timedelta(0):
        raise ValueError("Cutoff must be in UTC.")
    known = {}
    for result in results:
        if result.available_at <= cutoff:
            if result.match_id in known:
                raise ValueError(f"Duplicate original result for {result.match_id}.")
            known[result.match_id] = result
    seen = {}
    for correction in sorted(corrections, key=lambda row: row.available_at):
        if correction.available_at > cutoff:
            break
        match = known.get(correction.match_id)
        if (
            match is None
            or correction.winner_id not in (match.p1_id, match.p2_id)
            or correction.available_at < match.available_at
        ):
            raise ValueError(f"Invalid winner correction for {correction.match_id}.")
        key = (correction.match_id, correction.available_at)
        if key in seen:
            if seen[key] != correction.winner_id:
                raise ValueError(f"Conflicting corrections for {correction.match_id}.")
            continue
        if correction.available_at == match.available_at:
            raise ValueError(
                f"Correction must follow the original for {correction.match_id}."
            )
        seen[key] = correction.winner_id
        known[correction.match_id] = replace(
            match,
            label=int(correction.winner_id == match.p1_id),
            available_at=correction.available_at,
        )
    return sorted(known.values(), key=lambda row: (row.event_date, row.match_id))


def _elo_change(label: int, p1_elo: float, p2_elo: float) -> float:
    return K * (label - elo_win_probability(p1_elo, p2_elo))


def ratings_as_of(
    results: Iterable[MatchResult],
    cutoff: datetime,
    corrections: Iterable[WinnerCorrection] = (),
) -> dict[str, float]:
    """Rate visible results in day batches using each day's opening ratings."""
    ratings: dict[str, float] = {}
    for _, day in groupby(
        results_as_of(results, cutoff, corrections), key=lambda row: row.event_date
    ):
        changes: defaultdict[str, list[float]] = defaultdict(list)
        for match in day:
            p1 = ratings.get(match.p1_id, INITIAL_ELO)
            p2 = ratings.get(match.p2_id, INITIAL_ELO)
            change = _elo_change(match.label, p1, p2)
            changes[match.p1_id].append(change)
            changes[match.p2_id].append(-change)
        for player, deltas in changes.items():
            ratings[player] = ratings.get(player, INITIAL_ELO) + fsum(deltas)
    return ratings


@dataclass(frozen=True)
class MatchForecast:
    match_id: str
    cutoff: datetime
    p1_elo: float
    p2_elo: float

    @property
    def p1_win_probability(self) -> float:
        return elo_win_probability(self.p1_elo, self.p2_elo)


def leaky_elo_probability(forecast: MatchForecast, label: int) -> float:
    """Deliberately consume the current label before predicting it."""
    change = _elo_change(label, forecast.p1_elo, forecast.p2_elo)
    return elo_win_probability(forecast.p1_elo + change, forecast.p2_elo - change)


def prior_day_cutoff(event_date: date) -> datetime:
    return datetime.combine(event_date, time.min, timezone.utc) - timedelta(seconds=1)


def replay_matches(
    results: Sequence[MatchResult],
    corrections: Sequence[WinnerCorrection] = (),
) -> Iterator[tuple[MatchResult, MatchForecast]]:
    """Yield labels and pre-match features at the end of the prior UTC day."""
    # ponytail: rebuild once per day for this small extract, use checkpoints when this becomes costly.
    ordered = sorted(results, key=lambda row: (row.event_date, row.match_id))
    for event_date, day in groupby(ordered, key=lambda row: row.event_date):
        cutoff = prior_day_cutoff(event_date)
        ratings = ratings_as_of(ordered, cutoff, corrections)
        for match in day:
            yield (
                match,
                MatchForecast(
                    match.match_id,
                    cutoff,
                    ratings.get(match.p1_id, INITIAL_ELO),
                    ratings.get(match.p2_id, INITIAL_ELO),
                ),
            )


def development_scores(results: Sequence[MatchResult]) -> dict[str, int | float]:
    """Measure the direct effect of feeding the current label to Elo early."""
    development = [match for match in results if match.event_date < DEVELOPMENT_END]
    losses, leaked_losses = [], []
    for match, forecast in replay_matches(development):
        probability = forecast.p1_win_probability
        leaked = leaky_elo_probability(forecast, match.label)
        losses.append(-log(probability if match.label else 1 - probability))
        leaked_losses.append(-log(leaked if match.label else 1 - leaked))
    if not losses:
        raise ValueError("No matches in the development period.")
    return {
        "matches": len(losses),
        "as_of_log_loss": fsum(losses) / len(losses),
        "leaky_log_loss": fsum(leaked_losses) / len(leaked_losses),
    }


def worked_fixture() -> dict[str, object]:
    """Replay F01 and identify the result that informed the noon ratings."""
    cutoff = datetime.fromisoformat(F01_CUTOFF.replace("Z", "+00:00"))
    p1, p2 = F01_PLAYERS
    prior = MatchResult(
        "F01-prior",
        date(2030, 6, 9),
        cutoff - timedelta(hours=1),
        F01_SURFACE,
        p2,
        "P03",
        "Robin Novak",
        "New player",
        1,
    )
    # F01 really finished after noon on June 10. Its result enters Elo after the day closes.
    fixture_result = MatchResult(
        F01_ID,
        cutoff.date(),
        cutoff + timedelta(hours=12),
        F01_SURFACE,
        p1,
        p2,
        "Alex Silva",
        "Robin Novak",
        1,
    )
    correction = WinnerCorrection(prior.match_id, "P03", cutoff + timedelta(minutes=30))
    results = [prior, fixture_result]
    corrections = [correction]
    ratings = ratings_as_of(results, cutoff, corrections)
    forecast = MatchForecast(
        F01_ID, cutoff, ratings.get(p1, INITIAL_ELO), ratings.get(p2, INITIAL_ELO)
    )
    return {
        "fixture_id": F01_ID,
        "cutoff": F01_CUTOFF,
        "p1_id": p1,
        "p2_id": p2,
        "p1_elo": forecast.p1_elo,
        "p2_elo": forecast.p2_elo,
        "p1_win_probability": forecast.p1_win_probability,
        "contributing_results": [
            row.match_id for row in results_as_of(results, cutoff, corrections)
        ],
        "separate_correction": {
            **asdict(correction),
            "available_at": correction.available_at.isoformat(),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        nargs="?",
        help="Canonical Parquet for the development comparison. Omit to replay F01.",
    )
    args = parser.parse_args(argv)
    output = (
        development_scores(read_results(args.results))
        if args.results
        else worked_fixture()
    )
    print(json.dumps(output))


if __name__ == "__main__":
    main()
