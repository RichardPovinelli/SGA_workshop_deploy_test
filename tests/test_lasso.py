# pylint: disable=missing-module-docstring, missing-function-docstring

from __future__ import annotations

import numpy as np
import pytest

from src.models.lasso import evaluate_lasso_model, fit_lasso_model, predict_lasso


def test_lasso_fit_predict_runs_and_aligns_index(synthetic_model_df) -> None:
    model = fit_lasso_model(synthetic_model_df, 4)
    test_df = synthetic_model_df.loc[synthetic_model_df["split"] == "test"]
    predictions = predict_lasso(model, test_df)

    assert predictions.index.equals(test_df.index)
    assert not predictions.isna().any()


def test_lasso_scaler_uses_train_rows_only(synthetic_model_df) -> None:
    model = fit_lasso_model(synthetic_model_df, 1)
    scaler = model.estimator.named_steps["scaler"]
    train_df = synthetic_model_df.loc[
        synthetic_model_df["split"] == "train", list(model.feature_columns)
    ]

    assert np.allclose(scaler.mean_, train_df.mean().to_numpy())


def test_evaluate_lasso_model_returns_shared_results(synthetic_model_df) -> None:
    model, results = evaluate_lasso_model(synthetic_model_df, 5)

    assert model.horizon == 5
    assert model.alpha > 0.0
    assert results["zone"].tolist() == ["zone_a", "zone_b"]
    assert results["model"].tolist() == ["lasso", "lasso"]
    assert not results.isna().any().any()


def test_lasso_validates_horizon(synthetic_model_df) -> None:
    with pytest.raises(ValueError, match="Horizon must be one of"):
        fit_lasso_model(synthetic_model_df, 9)
