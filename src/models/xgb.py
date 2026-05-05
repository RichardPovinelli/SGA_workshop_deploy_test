"""Direct XGBoost baselines for workshop inference and comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import ALLOWED_HORIZONS
from src.data.split import (
    get_feature_columns,
    get_target_column,
    get_test_df,
    get_train_df,
    validate_horizon,
)
from src.evaluation.metrics import RESULTS_TABLE_COLUMNS, build_results_table


XGB_RESULTS_COLUMNS: tuple[str, ...] = RESULTS_TABLE_COLUMNS + ("n_train", "n_test")
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "n_estimators": 80,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": 1,
    "random_state": 0,
}


@dataclass(frozen=True)
class DirectXGBModel:
    """Fitted direct XGBoost model metadata for one horizon."""

    horizon: int
    feature_columns: tuple[str, ...]
    target_column: str
    estimator: Any


def _import_xgboost() -> Any:
    try:
        from xgboost import XGBRegressor  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required for Task 5 models. Install the 'xgboost' package "
            "in the workshop environment before calling src.models.xgb."
        ) from exc
    return XGBRegressor


def _make_xgb_results_table(  # pylint: disable=too-many-arguments
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    horizon: int,
    n_train: int,
    n_test: int,
) -> pd.DataFrame:
    results = build_results_table(
        y_true,
        y_pred,
        model_name="xgboost",
        horizon=horizon,
    )
    results["n_train"] = n_train
    results["n_test"] = n_test
    return results.loc[:, list(XGB_RESULTS_COLUMNS)]


def fit_xgb(
    df: pd.DataFrame,
    horizon: int,
    *,
    params: dict[str, Any] | None = None,
) -> DirectXGBModel:
    """Fit one direct horizon model using train rows only."""
    validated_horizon = validate_horizon(horizon)
    train_df = get_train_df(df)
    if train_df.empty:
        raise ValueError("No training rows available.")

    feature_columns = tuple(get_feature_columns(validated_horizon))
    target_column = get_target_column(validated_horizon)
    estimator_params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    estimator = _import_xgboost()(**estimator_params)
    estimator.fit(train_df.loc[:, feature_columns], train_df[target_column])

    return DirectXGBModel(
        horizon=validated_horizon,
        feature_columns=feature_columns,
        target_column=target_column,
        estimator=estimator,
    )


def predict_xgb(model: DirectXGBModel, df: pd.DataFrame) -> pd.Series:
    """Return predictions aligned to the input index."""
    predictions = model.estimator.predict(df.loc[:, list(model.feature_columns)])
    return pd.Series(predictions, index=df.index, name="prediction")


def generate_xgb_residuals(model: DirectXGBModel, df: pd.DataFrame) -> pd.DataFrame:
    """Return actuals, predictions, and residuals for diagnostics."""
    predictions = predict_xgb(model, df)
    actuals = df[model.target_column]
    residuals_df = pd.DataFrame(
        {
            "date": df["date"].to_numpy(copy=True),
            "horizon": model.horizon,
            "actual": actuals.to_numpy(copy=True),
            "prediction": predictions.to_numpy(copy=True),
        },
        index=df.index,
    )
    residuals_df["residual"] = residuals_df["actual"] - residuals_df["prediction"]
    return residuals_df


def evaluate_xgb(
    df: pd.DataFrame,
    horizon: int,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[DirectXGBModel, pd.DataFrame]:
    """Fit one horizon model on all data, predict on test split, and score it."""
    model = fit_xgb(df, horizon, params=params)
    train_df = get_train_df(df)
    test_df = get_test_df(df)
    if test_df.empty:
        raise ValueError("No test rows available.")

    predictions = predict_xgb(model, test_df)
    results = _make_xgb_results_table(
        test_df[model.target_column],
        predictions,
        horizon=model.horizon,
        n_train=len(train_df),
        n_test=len(test_df),
    )
    return model, results
