"""Residual diagnostic helpers for notebook-friendly reporting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from src.evaluation.metrics import _to_1d_array


def ljung_box_diagnostics(
    residuals: Sequence[float] | pd.Series | np.ndarray,
    lags: int | Sequence[int],
) -> pd.DataFrame:
    """Return Ljung-Box diagnostics as a clean DataFrame."""
    residual_array = _to_1d_array(residuals, name="residuals")
    diagnostics_df = acorr_ljungbox(residual_array, lags=lags, return_df=True)
    return diagnostics_df.rename(
        columns={"lb_stat": "statistic", "lb_pvalue": "p_value"}
    )
