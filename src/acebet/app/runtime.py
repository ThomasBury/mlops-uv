"""Application-owned runtime assets used by prediction routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from acebet.app.dependencies.data_models import ModelIdentity
from acebet.features import PredictionEstimator

PredictionMode = Literal["production", "testing"]


@dataclass(frozen=True, slots=True)
class PredictionAssets:
    """One complete set of assets required to serve a prediction mode."""

    model: PredictionEstimator
    player_stats: pd.DataFrame
    model_identity: ModelIdentity


@dataclass(slots=True)
class RuntimeAssets:
    """Mutable application container for production and testing assets."""

    production: PredictionAssets | None = None
    testing: PredictionAssets | None = None

    def lookup(self, prediction_mode: PredictionMode) -> PredictionAssets | None:
        """Return the assets for one mode, or ``None`` when unavailable."""
        return getattr(self, prediction_mode)

    def is_ready(self, *, require_testing: bool) -> bool:
        """Return whether every asset mode required by the environment is loaded."""
        return self.production is not None and (
            self.testing is not None or not require_testing
        )

    def clear(self) -> None:
        """Release references to all loaded runtime assets."""
        self.production = None
        self.testing = None
