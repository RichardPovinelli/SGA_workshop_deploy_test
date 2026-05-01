"""Naive direct-forecast baseline."""

from __future__ import annotations

import pandas as pd

from src.data.split import get_target_column, validate_horizon
from src.evaluation.metrics import build_results_table


def predict_naive(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Return a workshop-friendly persistence forecast aligned to ``df``.

    The rule is intentionally simple and deterministic:
    - horizon 1 uses ``lag1`` (yesterday's demand)
    - horizons 2 through 7 use ``lag7`` (same weekday from the prior week)
    """
    validated_horizon = validate_horizon(horizon)
    source_column = "lag1" if validated_horizon == 1 else "lag7"
    return pd.Series(
        df[source_column].to_numpy(copy=True), index=df.index, name="prediction"
    )


def evaluate_naive_model(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return the shared per-zone evaluation table for the naive baseline."""
    validated_horizon = validate_horizon(horizon)
    predictions = predict_naive(df, validated_horizon)
    target_column = get_target_column(validated_horizon)
    return build_results_table(
        df.loc[:, ["zone"]],
        df[target_column],
        predictions,
        model_name="naive",
        horizon=validated_horizon,
    )
