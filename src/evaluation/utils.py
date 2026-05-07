"""Utilities for model evaluation and metrics calculation."""

from math import floor, log10
import pandas as pd


def round_to_sig_figs(x: float, sig: int = 3) -> float:
    """Round a number to a specified number of significant figures.

    Parameters
    ----------
    x : float
        Value to round
    sig : int
        Number of significant figures (default: 3)

    Returns
    -------
    float
        Rounded value
    """
    if x == 0:
        return 0
    return round(x, -int(floor(log10(abs(x)))) + (sig - 1))


def round_metric_columns(
    df: pd.DataFrame,
    columns: list[str],
    sig: int = 3,
) -> pd.DataFrame:
    """Round metric columns in a dataframe to significant figures.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with metric columns
    columns : list[str]
        Column names to round (e.g., ['mape', 'wmape', 'rmse'])
    sig : int
        Number of significant figures (default: 3)

    Returns
    -------
    pd.DataFrame
        Dataframe with rounded metric columns
    """
    df_rounded = df.copy()
    for col in columns:
        if col in df_rounded.columns:
            df_rounded[col] = df_rounded[col].apply(lambda x: round_to_sig_figs(x, sig))
    return df_rounded
