"""Direct lasso-regression baseline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.split import (
    get_feature_columns,
    get_target_column,
    get_test_df,
    get_train_df,
    validate_horizon,
)
from src.evaluation.metrics import build_results_table


DEFAULT_LASSO_ALPHA = 0.1


@dataclass(frozen=True)
class DirectLassoModel:
    """Bundle the fitted estimator and contract metadata for one horizon."""

    horizon: int
    feature_columns: tuple[str, ...]
    target_column: str
    estimator: Pipeline
    alpha: float


def fit_lasso_model(
    df: pd.DataFrame,
    horizon: int,
    *,
    alpha: float = DEFAULT_LASSO_ALPHA,
) -> DirectLassoModel:
    """Fit a deterministic direct lasso baseline using training rows only."""
    validated_horizon = validate_horizon(horizon)
    feature_columns = tuple(get_feature_columns())
    target_column = get_target_column(validated_horizon)
    train_df = get_train_df(df)
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                Lasso(alpha=alpha, max_iter=10_000, random_state=0),
            ),
        ]
    )
    estimator.fit(train_df.loc[:, feature_columns], train_df[target_column])
    return DirectLassoModel(
        horizon=validated_horizon,
        feature_columns=feature_columns,
        target_column=target_column,
        estimator=estimator,
        alpha=alpha,
    )


def predict_lasso(model: DirectLassoModel, df: pd.DataFrame) -> pd.Series:
    """Return predictions aligned to the input index."""
    predictions = model.estimator.predict(df.loc[:, list(model.feature_columns)])
    return pd.Series(predictions, index=df.index, name="prediction")


def evaluate_lasso_model(
    df: pd.DataFrame,
    horizon: int,
    *,
    alpha: float = DEFAULT_LASSO_ALPHA,
) -> tuple[DirectLassoModel, pd.DataFrame]:
    """Fit on the train split, predict on the test split, and score by zone."""
    model = fit_lasso_model(df, horizon, alpha=alpha)
    test_df = get_test_df(df)
    predictions = predict_lasso(model, test_df)
    results = build_results_table(
        test_df.loc[:, ["zone"]],
        test_df[model.target_column],
        predictions,
        model_name="lasso",
        horizon=model.horizon,
    )
    return model, results
