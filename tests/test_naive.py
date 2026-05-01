# pylint: disable=missing-module-docstring, missing-function-docstring

from __future__ import annotations

import pytest

from src.models.naive import evaluate_naive_model, predict_naive


def test_predict_naive_uses_lag1_for_horizon_1(synthetic_model_df) -> None:
    predictions = predict_naive(synthetic_model_df, 1)

    assert predictions.index.equals(synthetic_model_df.index)
    assert predictions.equals(synthetic_model_df["lag1"].rename("prediction"))


def test_predict_naive_uses_lag7_for_higher_horizons(synthetic_model_df) -> None:
    predictions = predict_naive(synthetic_model_df, 4)

    assert predictions.equals(synthetic_model_df["lag7"].rename("prediction"))
    assert not predictions.isna().any()


def test_predict_naive_validates_horizon(synthetic_model_df) -> None:
    with pytest.raises(ValueError, match="Horizon must be one of"):
        predict_naive(synthetic_model_df, 8)


def test_evaluate_naive_model_returns_shared_results(synthetic_model_df) -> None:
    test_df = synthetic_model_df.loc[synthetic_model_df["split"] == "test"]
    results = evaluate_naive_model(test_df, 2)

    assert results["zone"].tolist() == ["zone_a", "zone_b"]
    assert results["model"].tolist() == ["naive", "naive"]
