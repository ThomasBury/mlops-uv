"""Promote a local exported model bundle for serving."""

from __future__ import annotations

import argparse

from datetime import datetime, timezone
from pathlib import Path

from acebet.model_artifacts import (
    PROMOTION_RECORD_SCHEMA_VERSION,
    build_promotion_record,
    promotion_record_path,
    resolve_latest_model_path,
    write_json,
)
from acebet.train.train import DEFAULT_OUTPUT_DIR


def promote_model(
    model_path: Path | None = None, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> Path:
    """Promote one exported model bundle and write the local promotion record."""
    target_model_path = model_path or resolve_latest_model_path(output_dir)
    target_model_path = target_model_path.resolve()
    record = build_promotion_record(target_model_path)
    record["schema_version"] = PROMOTION_RECORD_SCHEMA_VERSION
    record["promoted_at_utc"] = datetime.now(timezone.utc).isoformat()
    output_path = promotion_record_path(target_model_path.parent)
    write_json(output_path, record)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model promotion."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Promote a local AceBet model bundle for serving.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a specific model_*.joblib artifact to promote. Defaults to the latest model in the output directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory containing exported model bundles.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for local model promotion."""
    args = parse_args()
    target_model_path = Path(args.model_path) if args.model_path else None
    record_path = promote_model(
        model_path=target_model_path,
        output_dir=Path(args.output_dir),
    )
    print(f"Promoted model record: {record_path}")


if __name__ == "__main__":
    main()
