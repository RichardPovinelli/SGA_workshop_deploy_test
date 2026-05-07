"""Reusable Plotly-based visualization functions for workshop notebook."""

from typing import Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

from src.visualization.colors import ColorType, get_model_colors


def _get_colorblind_palette() -> dict[str, str]:
    """
    Return Paul Tol's 8-color colorblind-friendly palette.

    Designed to be distinguishable for:
    - Normal color vision
    - Deuteranopia (red-green colorblind)
    - Protanopia (red-green colorblind)
    - Tritanopia (blue-yellow colorblind)

    Reference: https://personal.sron.nl/~pault/colourschemes.pdf
    """
    return {
        "teal": "#1B9E77",
        "orange": "#D95F02",
        "purple": "#7570B3",
        "magenta": "#E7298A",
        "green": "#66A61E",
        "yellow": "#E6AB02",
        "brown": "#A6761D",
        "gray": "#666666",
    }


def _get_model_color(model_name: str) -> str:
    """
    Get colorblind-friendly color for a model name.

    Converts string model names to ColorType enum and delegates to get_model_colors().
    Handles common aliases (e.g., 'xgb' -> 'xgboost').

    Uses consistent mapping:
    - Baseline models: naive (teal), linear (orange)
    - Regularized models: lasso (purple), elasticnet (magenta)
    - ML models: xgboost (green), mlp_forecast (yellow)
    - Fallback: black for unrecognized models
    """
    # Normalize model name and handle aliases
    model_str = str(model_name).lower().strip()
    alias_map = {
        "xgb": "xgboost",
        "mlp": "mlp_forecast",
    }
    model_str = alias_map.get(model_str, model_str)

    # Try to find matching ColorType enum
    for color_type in ColorType:
        if color_type.value == model_str:
            return get_model_colors(color_type)

    # Fallback: return black for unrecognized models
    return "#000000"


def _get_theme():
    """Return consistent Plotly theme configuration."""
    return {
        "font": {"family": "Arial, sans-serif", "size": 11},
        "margin": {"l": 80, "r": 50, "t": 80, "b": 60},
        "hovermode": "x unified",
    }


def plot_time_series_with_predictions(
    actual: pd.Series,
    predictions: pd.Series,
    title: str,
    zone: str = "",
) -> go.Figure:
    """
    Create overlaid time series plot of actual vs predicted values.

    Parameters
    ----------
    actual : pd.Series
        Actual values (indexed by dates or sequential)
    predictions : pd.Series
        Predicted values (must align with actual)
    title : str
        Plot title
    zone : str
        Zone identifier for display (optional)

    Returns
    -------
    go.Figure
        Plotly figure with overlaid time series
    """
    fig = go.Figure()

    # Extract x-axis values - prefer dates from index if available
    if pd.api.types.is_datetime64_any_dtype(actual.index):
        x_vals = actual.index.strftime("%Y-%m-%d").tolist()
        hover_format = "%{x}: %{y:.2f}"
        x_axis_label = "Date"
    else:
        x_vals = list(range(len(actual)))
        hover_format = "Step %{x}: %{y:.2f}"
        x_axis_label = "Time Step"

    palette = _get_colorblind_palette()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=actual.values,
            mode="lines",
            name="Actual",
            line={"color": palette["teal"], "width": 2},
            hovertemplate=f"{hover_format}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=predictions.values,
            mode="lines",
            name="Predicted",
            line={"color": palette["orange"], "width": 2, "dash": "dash"},
            hovertemplate=f"{hover_format}<extra></extra>",
        )
    )

    zone_label = f" (Zone: {zone})" if zone else ""
    fig.update_layout(
        title=f"{title}{zone_label}",
        xaxis_title=x_axis_label,
        yaxis_title="Demand (MW)",
        template="plotly_white",
        **_get_theme(),
    )

    return fig


def plot_model_metrics_comparison(
    metrics_df: pd.DataFrame,
    metric_col: str,
    title: str,
    models: Optional[list[str]] = None,
) -> go.Figure:
    """
    Create grouped bar chart comparing metrics across models and horizons.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Data with columns: horizon, model, and metric_col
    metric_col : str
        Column name for metric to plot (e.g., 'rmse', 'mae')
    title : str
        Plot title
    models : list[str], optional
        Filter to specific models (default: use all)

    Returns
    -------
    go.Figure
        Plotly grouped bar chart
    """
    if models:
        plot_df = metrics_df[metrics_df.get("model", metrics_df.get("Model", "")).isin(models)]
    else:
        plot_df = metrics_df.copy()

    # Pivot for grouped bar chart
    pivot_df = plot_df.pivot_table(
        index="horizon" if "horizon" in plot_df.columns else "Horizon",
        columns="model" if "model" in plot_df.columns else "Model",
        values=metric_col,
        aggfunc="first",
    )

    fig = go.Figure()
    for col in pivot_df.columns:
        fig.add_trace(
            go.Bar(
                x=pivot_df.index,
                y=pivot_df[col],
                name=str(col),
                marker=dict(color=_get_model_color(col)),
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Horizon",
        yaxis_title=metric_col.upper(),
        barmode="group",
        template="plotly_white",
        **_get_theme(),
    )

    return fig


def plot_horizon_degradation(
    results_by_horizon: pd.DataFrame,
    metric_col: str = "rmse",
    title: str = "Forecast Performance by Horizon",
) -> go.Figure:
    """
    Create line plot showing metric degradation across forecast horizons.

    Parameters
    ----------
    results_by_horizon : pd.DataFrame
        Data with columns: horizon, metric_col, and optionally model
    metric_col : str
        Column name for metric (default: 'rmse')
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly line plot with markers
    """
    fig = go.Figure()

    # Check if we have model-level data
    if "model" in results_by_horizon.columns:
        models = results_by_horizon["model"].unique()
        for model in sorted(models):
            model_data = results_by_horizon[results_by_horizon["model"] == model]
            model_data = model_data.sort_values("horizon")
            fig.add_trace(
                go.Scatter(
                    x=model_data["horizon"],
                    y=model_data[metric_col],
                    mode="lines+markers",
                    name=f"Model {model}",
                    hovertemplate="Horizon %{x}: %{y:.4f}<extra></extra>",
                )
            )
    else:
        results_by_horizon = results_by_horizon.sort_values("horizon")
        fig.add_trace(
            go.Scatter(
                x=results_by_horizon["horizon"],
                y=results_by_horizon[metric_col],
                mode="lines+markers",
                name=metric_col.upper(),
                line={"width": 3},
                marker={"size": 8},
                hovertemplate="Horizon %{x}: %{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Forecast Horizon (steps ahead)",
        yaxis_title=metric_col.upper(),
        template="plotly_white",
        **_get_theme(),
    )

    return fig


def plot_residuals_diagnostics(
    residuals: pd.Series,
    title: str = "Residual Diagnostics",
) -> go.Figure:
    """
    Create diagnostic subplots for residual analysis: histogram, Q-Q plot, and ACF.

    Parameters
    ----------
    residuals : pd.Series
        Residuals from model predictions
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly subplots with histogram, Q-Q, and ACF
    """
    residuals = residuals.dropna()

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Residual Distribution", "Q-Q Plot", "Sample ACF"),
    )

    # Histogram of residuals
    palette = _get_colorblind_palette()
    fig.add_trace(
        go.Histogram(
            x=residuals,
            name="Residuals",
            nbinsx=30,
            marker_color=palette["teal"],
            hovertemplate="%{x:.3f}: %{y} samples<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Q-Q Plot
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
    sample_quantiles = np.quantile(residuals, np.linspace(0.01, 0.99, len(sorted_residuals)))

    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode="markers",
            name="Q-Q",
            marker={"color": palette["orange"], "size": 4},
            hovertemplate="Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Add 45-degree reference line for Q-Q
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Reference",
            line={"color": "gray", "dash": "dash"},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )

    # ACF (autocorrelation) - simple implementation
    max_lag = min(20, len(residuals) // 4)
    acf_values = [residuals.autocorr(lag=i) for i in range(1, max_lag + 1)]
    lags = list(range(1, max_lag + 1))

    fig.add_trace(
        go.Bar(
            x=lags,
            y=acf_values,
            name="ACF",
            marker_color=palette["green"],
            hovertemplate="Lag %{x}: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    # Add significance bands for ACF
    significance_level = 1.96 / np.sqrt(len(residuals))
    fig.add_hline(
        y=significance_level,
        line_dash="dash",
        line_color="red",
        row=1,
        col=3,
        annotation_text="Significance",
    )
    fig.add_hline(
        y=-significance_level,
        line_dash="dash",
        line_color="red",
        row=1,
        col=3,
    )

    fig.update_xaxes(title_text="Residuals", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)

    fig.update_xaxes(title_text="Lag", row=1, col=3)
    fig.update_yaxes(title_text="Autocorrelation", row=1, col=3)

    fig.update_layout(
        title=title,
        height=400,
        template="plotly_white",
        showlegend=False,
        **_get_theme(),
    )

    return fig


def plot_data_overview(
    df: pd.DataFrame,
    height: int = 500,
) -> go.Figure:
    """
    Create exploratory data overview: time series for single zone with demand and temperature.

    Parameters
    ----------
    df : pd.DataFrame
        Single zone dataset with columns: date, demand, temp, and other features
    height : int
        Height of the plot in pixels (default: 500)

    Returns
    -------
    go.Figure
        Plotly time series plot with dual y-axes
    """
    target_col = "demand" if "demand" in df.columns else "target"
    date_col = "date" if "date" in df.columns else None
    temp_col = "temp" if "temp" in df.columns else None

    x_vals = df[date_col] if date_col else list(range(len(df)))

    fig = go.Figure()

    # Demand trace (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=df[target_col],
            mode="lines",
            name="Demand",
            line={"width": 1},
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Demand: %{y:.2f}<extra></extra>"
            if date_col
            else "Time: %{x}<br>Demand: %{y:.2f}<extra></extra>",
        )
    )

    # Temperature trace (right y-axis) if available
    if temp_col and temp_col in df.columns:
        palette = _get_colorblind_palette()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=df[temp_col],
                mode="lines",
                name="Temperature",
                line={"width": 1, "color": palette["orange"]},
                yaxis="y2",
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Temperature: %{y:.2f}°F<extra></extra>"
                if date_col
                else "Time: %{x}<br>Temperature: %{y:.2f}°F<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Date" if date_col else "Time",
        yaxis_title="Demand (MW)",
        yaxis2=dict(
            title="Temperature (°F)",
            overlaying="y",
            side="right",
        )
        if temp_col and temp_col in df.columns
        else None,
        height=height,
        template="plotly_white",
        showlegend=True,
        **_get_theme(),
    )

    return fig


def plot_demand_vs_temperature(
    df: pd.DataFrame,
    height: int = 500,
) -> go.Figure:
    """
    Create scatter plot of Demand vs Temperature for training data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with columns: demand (or target), temp, date, split
    height : int
        Height of the plot in pixels (default: 500)

    Returns
    -------
    go.Figure
        Plotly scatter plot showing training data only
    """
    target_col = "demand" if "demand" in df.columns else "target"
    temp_col = "temp" if "temp" in df.columns else None
    split_col = "split" if "split" in df.columns else None
    date_col = "date" if "date" in df.columns else None

    if not temp_col or temp_col not in df.columns:
        raise ValueError("Dataset must contain 'temp' column")
    if not date_col or date_col not in df.columns:
        raise ValueError("Dataset must contain 'date' column")

    # Filter for training data only
    if split_col and split_col in df.columns:
        subset = df[df[split_col] == "train"]
    else:
        subset = df

    fig = go.Figure()

    palette = _get_colorblind_palette()
    fig.add_trace(
        go.Scatter(
            x=subset[temp_col],
            y=subset[target_col],
            mode="markers",
            name="Training Data",
            marker={"size": 6, "color": palette["teal"], "opacity": 0.6},
            customdata=subset[date_col],
            hovertemplate="Date: %{customdata|%Y-%m-%d}<br>Temperature: %{x:.2f}°F<br>Demand: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Demand vs Temperature",
        xaxis_title="Temperature (°F)",
        yaxis_title="Demand (MW)",
        height=height,
        template="plotly_white",
        showlegend=False,
        **_get_theme(),
    )

    return fig


def plot_data_split_distribution(
    split_counts: dict[str, int],
    title: str = "Data Split Distribution",
) -> go.Figure:
    """
    Create bar chart showing train/test/validation split proportions.

    Parameters
    ----------
    split_counts : dict[str, int]
        Mapping of split name to observation count (e.g., {'train': 1000, 'test': 200})
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly bar chart with split proportions
    """
    splits = list(split_counts.keys())
    counts = list(split_counts.values())
    total = sum(counts)
    percentages = [100 * c / total for c in counts]

    palette = _get_colorblind_palette()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=splits,
            y=counts,
            text=[f"{pct:.1f}%" for pct in percentages],
            textposition="outside",
            marker_color=[palette["teal"], palette["orange"], palette["green"]],
            hovertemplate="%{x}: %{y} observations<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Data Split",
        yaxis_title="Number of Observations",
        template="plotly_white",
        **_get_theme(),
    )

    return fig
