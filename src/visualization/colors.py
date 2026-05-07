"""Color definitions and management for visualization.

This module provides a centralized location for all color definitions used in
plots and visualizations. Colors are based on the Paul Tol colorblind-friendly
palette, designed to be distinguishable for all types of color blindness
(protanopia, deuteranopia, tritanopia).

Reference: https://personal.sron.nl/~pault/colourschemes.pdf
"""

from enum import Enum


class ColorType(Enum):
    """Enumeration of model and element types for color mapping.

    Baseline models:
    - NAIVE: Naive/persistence baseline
    - LINEAR: Linear regression baseline

    Regularized baseline models:
    - LASSO: L1-regularized linear model
    - ELASTICNET: Elastic Net regularized model

    Advanced ML models:
    - XGBOOST: Gradient boosted trees
    - MLP_FORECAST: Multi-layer perceptron forecaster
    - TFT: Temporal Fusion Transformer

    Visualization elements:
    - ACTUAL: Actual/observed values
    """

    # Baseline models
    NAIVE = "naive"
    LINEAR = "linear"

    # Regularized baselines
    LASSO = "lasso"
    ELASTICNET = "elasticnet"

    # ML models
    XGBOOST = "xgboost"
    MLP_FORECAST = "mlp_forecast"
    TFT = "tft"

    # Visualization elements
    ACTUAL = "actual"


def get_model_colors(color_type: ColorType) -> str:
    """Get colorblind-friendly color for a model or element type.

    Uses the Paul Tol 8-color palette, carefully designed to be distinguishable
    for people with color blindness (protanopia, deuteranopia, tritanopia).

    Color mapping:
    - Baseline models: teal (naive) and orange (linear)
    - Regularized baselines: purple (lasso) and magenta (elasticnet)
    - Advanced ML: green (xgboost) and yellow (mlp_forecast)
    - Visualization elements: teal (actual values)

    Parameters
    ----------
    color_type : ColorType
        The model or element type to get a color for

    Returns
    -------
    str
        Hex color code (e.g., "#1B9E77")
        Returns "#000000" (black) if color_type is not in the mapping

    Examples
    --------
    >>> get_model_colors(ColorType.XGBOOST)
    '#66A61E'
    >>> get_model_colors(ColorType.LINEAR)
    '#D95F02'
    """
    # Paul Tol 8-color palette for colorblind-friendly visualization
    _palette = {
        "teal": "#1B9E77",
        "orange": "#D95F02",
        "purple": "#7570B3",
        "magenta": "#E7298A",
        "green": "#66A61E",
        "yellow": "#E6AB02",
        "brown": "#A6761D",
        "gray": "#666666",
    }

    # Model-to-color mapping
    _color_mapping = {
        ColorType.NAIVE: _palette["teal"],
        ColorType.LINEAR: _palette["orange"],
        ColorType.LASSO: _palette["purple"],
        ColorType.ELASTICNET: _palette["magenta"],
        ColorType.XGBOOST: _palette["green"],
        ColorType.MLP_FORECAST: _palette["yellow"],
        ColorType.TFT: _palette["brown"],
        ColorType.ACTUAL: _palette["teal"],
    }

    return _color_mapping.get(color_type, "#000000")
