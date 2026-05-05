"""Metric helpers and shared results-table utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


RESULTS_TABLE_COLUMNS: tuple[str, ...] = (
    "horizon",
    "model",
    "mape",
    "wmape",
    "rmse",
)


def _to_1d_array(values: Sequence[float] | pd.Series | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _validate_paired_arrays(
    y_true: Sequence[float] | pd.Series | np.ndarray,
    y_pred: Sequence[float] | pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual = _to_1d_array(y_true, name="y_true")
    predicted = _to_1d_array(y_pred, name="y_pred")
    if actual.shape[0] != predicted.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")
    return actual, predicted


def mape(
    y_true: Sequence[float] | pd.Series | np.ndarray,
    y_pred: Sequence[float] | pd.Series | np.ndarray,
) -> float:
    """Return mean absolute percentage error while ignoring zero actuals."""
    actual, predicted = _validate_paired_arrays(y_true, y_pred)
    non_zero_mask = actual != 0.0
    if not np.any(non_zero_mask):
        raise ValueError("MAPE is undefined when all actual values are zero.")
    percentage_errors = np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])
    return float(np.mean(percentage_errors) * 100.0)


def wmape(
    y_true: Sequence[float] | pd.Series | np.ndarray,
    y_pred: Sequence[float] | pd.Series | np.ndarray,
) -> float:
    """Return weighted mean absolute percentage error."""
    actual, predicted = _validate_paired_arrays(y_true, y_pred)
    denominator = float(np.sum(np.abs(actual)))
    if denominator == 0.0:
        raise ValueError("WMAPE is undefined when the sum of absolute actuals is zero.")
    numerator = float(np.sum(np.abs(actual - predicted)))
    return float((numerator / denominator) * 100.0)


def rmse(
    y_true: Sequence[float] | pd.Series | np.ndarray,
    y_pred: Sequence[float] | pd.Series | np.ndarray,
) -> float:
    """Return root mean squared error."""
    actual, predicted = _validate_paired_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def build_results_table(
    y_true: Sequence[float] | pd.Series | np.ndarray,
    y_pred: Sequence[float] | pd.Series | np.ndarray,
    *,
    model_name: str,
    horizon: int,
) -> pd.DataFrame:
    """Return the aggregated results-table contract."""
    actual, predicted = _validate_paired_arrays(y_true, y_pred)

    results_row = {
        "horizon": horizon,
        "model": model_name,
        "mape": mape(actual, predicted),
        "wmape": wmape(actual, predicted),
        "rmse": rmse(actual, predicted),
    }

    return pd.DataFrame([results_row], columns=list(RESULTS_TABLE_COLUMNS))
