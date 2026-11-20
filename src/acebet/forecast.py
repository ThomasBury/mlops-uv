"""Replay the tutorial's pre-match baseline without loading model assets."""

import argparse
import json

F01_ID = "F01"
F01_CUTOFF = "2030-06-10T12:00:00Z"
F01_PLAYERS = ("P01", "P02")


def probability_for_first_player(
    first_player_id: str, second_player_id: str, canonical_probability: float
) -> float:
    """Express the smaller player ID's win probability in the requested order."""
    if not first_player_id.strip() or not second_player_id.strip():
        raise ValueError("Player IDs must be nonempty.")
    if first_player_id == second_player_id:
        raise ValueError("Player IDs must be distinct.")
    if not 0.0 <= canonical_probability <= 1.0:
        raise ValueError("Win probability must be between 0 and 1 inclusive.")
    if first_player_id < second_player_id:
        return canonical_probability
    return 1.0 - canonical_probability


def main() -> None:
    """Print the synthetic F01 fixture's baseline forecast."""
    argparse.ArgumentParser(description=main.__doc__).parse_args()
    first_player_id, second_player_id = F01_PLAYERS
    canonical_probability = 0.50
    print(
        json.dumps(
            {
                "fixture_id": F01_ID,
                "cutoff": F01_CUTOFF,
                "p1_id": first_player_id,
                "p2_id": second_player_id,
                "p1_win_probability": probability_for_first_player(
                    first_player_id, second_player_id, canonical_probability
                ),
            }
        )
    )
