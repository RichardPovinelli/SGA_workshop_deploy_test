"""Direct XGBoost baselines for workshop inference and comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import ALLOWED_HORIZONS
from src.data.schema import get_sorted_zones
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
    """Fitted direct XGBoost model metadata for one zone and one horizon."""

    zone: str
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


def _get_zone_df(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    zone_df = df.loc[df["zone"] == zone].copy()
    if zone_df.empty:
        raise ValueError(f"Zone {zone!r} was not found in the provided DataFrame.")
    return zone_df


def _make_xgb_results_table(  # pylint: disable=too-many-arguments
    metadata: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    horizon: int,
    n_train: int,
    n_test: int,
) -> pd.DataFrame:
    results = build_results_table(
        metadata,
        y_true,
        y_pred,
        model_name="xgboost",
        horizon=horizon,
    )
    results["n_train"] = n_train
    results["n_test"] = n_test
    return results.loc[:, list(XGB_RESULTS_COLUMNS)]


def fit_xgb_for_zone(
    df: pd.DataFrame,
    horizon: int,
    zone: str,
    *,
    params: dict[str, Any] | None = None,
) -> DirectXGBModel:
    """Fit one direct horizon model for one zone using train rows only."""
    validated_horizon = validate_horizon(horizon)
    zone_df = _get_zone_df(df, zone)
    train_df = get_train_df(zone_df)
    if train_df.empty:
        raise ValueError(f"Zone {zone!r} has no training rows.")

    feature_columns = tuple(get_feature_columns())
    target_column = get_target_column(validated_horizon)
    estimator_params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    estimator = _import_xgboost()(**estimator_params)
    estimator.fit(train_df.loc[:, feature_columns], train_df[target_column])

    return DirectXGBModel(
        zone=zone,
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
            "zone": df["zone"].astype(str).to_numpy(copy=True),
            "horizon": model.horizon,
            "actual": actuals.to_numpy(copy=True),
            "prediction": predictions.to_numpy(copy=True),
        },
        index=df.index,
    )
    residuals_df["residual"] = residuals_df["actual"] - residuals_df["prediction"]
    return residuals_df


def evaluate_xgb_for_zone(
    df: pd.DataFrame,
    horizon: int,
    zone: str,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[DirectXGBModel, pd.DataFrame]:
    """Fit one zone/horizon model, predict on that zone's test rows, and score it."""
    model = fit_xgb_for_zone(df, horizon, zone, params=params)
    zone_df = _get_zone_df(df, zone)
    train_df = get_train_df(zone_df)
    test_df = get_test_df(zone_df)
    if test_df.empty:
        raise ValueError(f"Zone {zone!r} has no test rows.")

    predictions = predict_xgb(model, test_df)
    results = _make_xgb_results_table(
        test_df.loc[:, ["zone"]],
        test_df[model.target_column],
        predictions,
        horizon=model.horizon,
        n_train=len(train_df),
        n_test=len(test_df),
    )
    return model, results


def fit_all_horizons_for_zone(
    df: pd.DataFrame,
    zone: str,
    *,
    params: dict[str, Any] | None = None,
) -> dict[int, DirectXGBModel]:
    """Fit horizons 1 through 7 for one selected zone."""
    return {
        horizon: fit_xgb_for_zone(df, horizon, zone, params=params)
        for horizon in ALLOWED_HORIZONS
    }


def fit_xgb_for_all_zones(
    df: pd.DataFrame,
    horizon: int,
    *,
    params: dict[str, Any] | None = None,
) -> dict[str, DirectXGBModel]:
    """Fit one horizon across all zones by looping zone-by-zone."""
    validated_horizon = validate_horizon(horizon)
    return {
        zone: fit_xgb_for_zone(df, validated_horizon, zone, params=params)
        for zone in get_sorted_zones(df)
    }


def evaluate_xgb_for_all_zones(
    df: pd.DataFrame,
    horizon: int,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[dict[str, DirectXGBModel], pd.DataFrame]:
    """Fit one horizon for all zones and return a combined results table."""
    models = fit_xgb_for_all_zones(df, horizon, params=params)
    results_frames: list[pd.DataFrame] = []

    for zone in get_sorted_zones(df):
        zone_df = _get_zone_df(df, zone)
        train_df = get_train_df(zone_df)
        test_df = get_test_df(zone_df)
        if test_df.empty:
            raise ValueError(f"Zone {zone!r} has no test rows.")

        model = models[zone]
        predictions = predict_xgb(model, test_df)
        results_frames.append(
            _make_xgb_results_table(
                test_df.loc[:, ["zone"]],
                test_df[model.target_column],
                predictions,
                horizon=model.horizon,
                n_train=len(train_df),
                n_test=len(test_df),
            )
        )

    combined_results = pd.concat(results_frames, ignore_index=True)
    return models, combined_results.loc[:, list(XGB_RESULTS_COLUMNS)]
