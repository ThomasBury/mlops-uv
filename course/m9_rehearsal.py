"""Build an alternate pair and labeled synthetic decisions for the M9 lab."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from shutil import copyfile, copytree

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from acebet.dataprep.historical_results import content_sha256, read_results
from acebet.model_artifacts import read_json, sha256_file, write_json
from acebet.train.historical_evaluation import evaluate_candidate, train_candidate


def delayed_run(history: Path, output: Path) -> None:
    """Delay one archived result to make a different, internally compatible pair."""
    results = read_results(history)
    results[0] = replace(results[0], available_at=results[-1].available_at)
    schema = pq.read_schema(history)
    metadata = {
        **(schema.metadata or {}),
        b"content_sha256": content_sha256(results).encode(),
    }
    table = pa.Table.from_pylist(
        [asdict(match) for match in results], schema=schema.with_metadata(metadata)
    )
    output.mkdir()
    path = output / "matches.parquet"
    pq.write_table(table, path)
    bundle = output / "bundle"
    train_candidate(read_results(path), bundle)
    evaluate_candidate(read_results(path), bundle)


def synthetic_scores(run: Path, output: Path) -> None:
    """Exercise the approval branch with a deliberately wrong comparator."""
    output.mkdir()
    copyfile(run / "matches.parquet", output / "matches.parquet")
    bundle = output / "bundle"
    copytree(run / "bundle", bundle)
    evaluation_path = bundle / "final/evaluation.json"
    original = read_json(evaluation_path)
    evaluation = {
        key: original[key]
        for key in ("selection_run_id", "selection_sha256", "model_sha256", "matches")
    }
    evaluation["rehearsal"] = {
        "reason": "Synthetic approval exercise. Elo scores were replaced by 1 - label.",
        "original_evaluation_sha256": sha256_file(evaluation_path),
    }
    forecasts_path = bundle / "final/forecasts.csv"
    forecasts = pd.read_csv(forecasts_path, dtype={"match_id": str})
    forecasts["elo"] = 1 - forecasts["label"]
    forecasts.to_csv(forecasts_path, index=False)
    evaluation["forecasts_sha256"] = sha256_file(forecasts_path)
    write_json(evaluation_path, evaluation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    delay = subcommands.add_parser("delay")
    delay.add_argument("history", type=Path)
    delay.add_argument("output", type=Path)
    scores = subcommands.add_parser("scores")
    scores.add_argument("run", type=Path)
    scores.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "delay":
        delayed_run(args.history, args.output)
    else:
        synthetic_scores(args.run, args.output)


if __name__ == "__main__":
    main()
