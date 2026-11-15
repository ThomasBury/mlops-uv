"""Importable estimators used by training and serialized serving artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from acebet.features import MODEL_FEATURE_COLUMNS

BINARY_TARGET_CLASSES = frozenset({0, 1})


def _require_binary_target_classes(values: np.ndarray, context: str) -> None:
    """Raise when a training or evaluation target is not binary and complete."""
    observed = set(np.unique(values).tolist())
    unexpected = observed - BINARY_TARGET_CLASSES
    if unexpected:
        raise ValueError(
            f"{context} target contains unexpected classes {sorted(unexpected)}; expected only [0, 1]."
        )
    missing = BINARY_TARGET_CLASSES - observed
    if missing:
        raise ValueError(
            f"{context} must contain both target classes [0, 1]; missing {sorted(missing)} and observed {sorted(observed)}."
        )


def _ordered_feature_frame(X: Any, context: str) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{context} requires a pandas DataFrame.")
    expected = list(MODEL_FEATURE_COLUMNS)
    observed = list(X.columns)
    if observed != expected:
        raise ValueError(
            f"{context} feature contract mismatch: expected ordered columns {expected}, found {observed}."
        )
    return X


class EloBaselineClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible classifier that serves the prepared ``proba_elo`` feature."""

    def fit(self, X: pd.DataFrame, y: Any) -> "EloBaselineClassifier":
        """Validate and record the fitted binary dataframe contract."""
        frame = _ordered_feature_frame(X, "EloBaselineClassifier.fit")
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(frame):
            raise ValueError(
                "EloBaselineClassifier.fit requires a one-dimensional target aligned with X."
            )
        _require_binary_target_classes(target, "Elo baseline training data")
        self.classes_ = np.asarray([0, 1], dtype=target.dtype)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return ``[1 - proba_elo, proba_elo]`` after strict probability checks."""
        check_is_fitted(self, ("classes_", "n_features_in_", "feature_names_in_"))
        frame = _ordered_feature_frame(X, "EloBaselineClassifier.predict_proba")
        try:
            probability = frame["proba_elo"].to_numpy(dtype=float, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("proba_elo values must be numeric.") from exc
        if not np.isfinite(probability).all():
            raise ValueError("proba_elo values must be finite.")
        if ((probability < 0) | (probability > 1)).any():
            raise ValueError("proba_elo values must be in [0, 1].")
        return np.column_stack((1.0 - probability, probability))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the class with highest probability, preferring class 0 on ties."""
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
