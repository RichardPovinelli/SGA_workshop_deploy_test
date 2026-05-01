# pylint: disable=missing-module-docstring, missing-function-docstring

from __future__ import annotations

import numpy as np
import pytest

from src.models.linear import evaluate_linear_model, fit_linear_model, predict_linear


def test_linear_fit_predict_runs_and_aligns_index(synthetic_model_df) -> None:
    model = fit_linear_model(synthetic_model_df, 3)
    test_df = synthetic_model_df.loc[synthetic_model_df["split"] == "test"]
    predictions = predict_linear(model, test_df)

    assert predictions.index.equals(test_df.index)
    assert not predictions.isna().any()


def test_linear_scaler_uses_train_rows_only(synthetic_model_df) -> None:
    model = fit_linear_model(synthetic_model_df, 1)
    scaler = model.estimator.named_steps["scaler"]
    train_df = synthetic_model_df.loc[
        synthetic_model_df["split"] == "train", list(model.feature_columns)
    ]

    assert np.allclose(scaler.mean_, train_df.mean().to_numpy())


def test_evaluate_linear_model_returns_shared_results(synthetic_model_df) -> None:
    model, results = evaluate_linear_model(synthetic_model_df, 2)

    assert model.horizon == 2
    assert results["zone"].tolist() == ["zone_a", "zone_b"]
    assert results["model"].tolist() == ["linear", "linear"]
    assert not results.isna().any().any()


def test_linear_validates_horizon(synthetic_model_df) -> None:
    with pytest.raises(ValueError, match="Horizon must be one of"):
        fit_linear_model(synthetic_model_df, 0)
