"""Lightweight Diebold-Mariano test wrapper."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from src.evaluation.metrics import _to_1d_array


SUPPORTED_LOSSES: tuple[str, ...] = ("absolute", "squared")


def _loss_series(
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    loss: str,
) -> np.ndarray:
    errors = actual - predicted
    if loss == "absolute":
        return np.abs(errors)
    if loss == "squared":
        return errors**2
    raise ValueError(f"loss must be one of {SUPPORTED_LOSSES}, got {loss!r}.")


def diebold_mariano_test(
    actuals: Sequence[float] | pd.Series | np.ndarray,
    predictions_a: Sequence[float] | pd.Series | np.ndarray,
    predictions_b: Sequence[float] | pd.Series | np.ndarray,
    *,
    loss: str = "squared",
) -> dict[str, float | int | str]:
    """Compare two forecast series with a simple paired-loss t-statistic.

    This workshop wrapper uses the mean loss differential divided by the
    standard error of that differential. It is intentionally lightweight and
    suitable for short notebook examples rather than a full HAC-corrected
    econometrics implementation.
    """
    actual = _to_1d_array(actuals, name="actuals")
    predicted_a = _to_1d_array(predictions_a, name="predictions_a")
    predicted_b = _to_1d_array(predictions_b, name="predictions_b")

    if (
        actual.shape[0] != predicted_a.shape[0]
        or actual.shape[0] != predicted_b.shape[0]
    ):
        raise ValueError(
            "actuals, predictions_a, and predictions_b must have the same length."
        )
    if actual.shape[0] < 2:
        raise ValueError("Diebold-Mariano test requires at least two observations.")

    loss_a = _loss_series(actual, predicted_a, loss=loss)
    loss_b = _loss_series(actual, predicted_b, loss=loss)
    differential = loss_a - loss_b
    n_obs = int(differential.shape[0])

    variance = float(np.var(differential, ddof=1))
    if variance == 0.0:
        statistic = 0.0
        p_value = 1.0
    else:
        standard_error = float(np.sqrt(variance / n_obs))
        statistic = float(np.mean(differential) / standard_error)
        p_value = float(student_t.sf(np.abs(statistic), df=n_obs - 1) * 2.0)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "n_obs": n_obs,
        "loss": loss,
    }
