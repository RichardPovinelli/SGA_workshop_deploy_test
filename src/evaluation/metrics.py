"""Metric helpers and shared results-table utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


RESULTS_TABLE_COLUMNS: tuple[str, ...] = (
    "zone",
    "horizon",
    "model",
    "mape",
    "wmape",
    "rmse",
)


def _to_1d_array(
    values: Sequence[float] | pd.Series | np.ndarray, *, name: str
) -> np.ndarray:
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
    percentage_errors = np.abs(
        (actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask]
    )
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
    metadata: pd.DataFrame,
    y_true: Sequence[float] | pd.Series | np.ndarray,
    y_pred: Sequence[float] | pd.Series | np.ndarray,
    *,
    model_name: str,
    horizon: int,
) -> pd.DataFrame:
    """Return the shared per-zone results-table contract."""
    actual, predicted = _validate_paired_arrays(y_true, y_pred)
    if len(metadata) != actual.shape[0]:
        raise ValueError(
            "metadata must have the same number of rows as y_true and y_pred."
        )
    if "zone" not in metadata.columns:
        raise ValueError("metadata must contain a 'zone' column.")

    results_rows: list[dict[str, object]] = []
    working_df = metadata.loc[:, ["zone"]].copy()
    working_df["y_true"] = actual
    working_df["y_pred"] = predicted

    for zone in sorted(working_df["zone"].astype(str).unique().tolist()):
        zone_df = working_df.loc[working_df["zone"] == zone]
        results_rows.append(
            {
                "zone": zone,
                "horizon": horizon,
                "model": model_name,
                "mape": mape(zone_df["y_true"], zone_df["y_pred"]),
                "wmape": wmape(zone_df["y_true"], zone_df["y_pred"]),
                "rmse": rmse(zone_df["y_true"], zone_df["y_pred"]),
            }
        )

    return pd.DataFrame(results_rows, columns=list(RESULTS_TABLE_COLUMNS))
