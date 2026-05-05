"""Direct ElasticNet regression baseline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import ElasticNet
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


DEFAULT_ELASTICNET_ALPHA = 0.1
DEFAULT_L1_RATIO = 0.5


@dataclass(frozen=True)
class DirectElasticNetModel:
    """Bundle the fitted estimator and contract metadata for one horizon."""

    horizon: int
    feature_columns: tuple[str, ...]
    target_column: str
    estimator: Pipeline
    alpha: float
    l1_ratio: float


def fit_elasticnet_model(
    df: pd.DataFrame,
    horizon: int,
    *,
    alpha: float = DEFAULT_ELASTICNET_ALPHA,
    l1_ratio: float = DEFAULT_L1_RATIO,
) -> DirectElasticNetModel:
    """Fit a deterministic direct ElasticNet baseline using training rows only."""
    validated_horizon = validate_horizon(horizon)
    feature_columns = tuple(get_feature_columns(validated_horizon))
    target_column = get_target_column(validated_horizon)
    train_df = get_train_df(df)
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10_000, random_state=0),
            ),
        ]
    )
    estimator.fit(train_df.loc[:, feature_columns], train_df[target_column])
    return DirectElasticNetModel(
        horizon=validated_horizon,
        feature_columns=feature_columns,
        target_column=target_column,
        estimator=estimator,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )


def predict_elasticnet(model: DirectElasticNetModel, df: pd.DataFrame) -> pd.Series:
    """Return predictions aligned to the input index."""
    predictions = model.estimator.predict(df.loc[:, list(model.feature_columns)])
    return pd.Series(predictions, index=df.index, name="prediction")


def evaluate_elasticnet_model(
    df: pd.DataFrame,
    horizon: int,
    *,
    alpha: float = DEFAULT_ELASTICNET_ALPHA,
    l1_ratio: float = DEFAULT_L1_RATIO,
) -> tuple[DirectElasticNetModel, pd.DataFrame]:
    """Fit on the train split, predict on the test split, and score aggregated."""
    model = fit_elasticnet_model(df, horizon, alpha=alpha, l1_ratio=l1_ratio)
    test_df = get_test_df(df)
    predictions = predict_elasticnet(model, test_df)
    results = build_results_table(
        test_df[model.target_column],
        predictions,
        model_name="elasticnet",
        horizon=model.horizon,
    )
    return model, results
