# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=duplicate-code

from __future__ import annotations

import pytest

from src.models.xgb import (
    XGB_RESULTS_COLUMNS,
    evaluate_xgb_for_all_zones,
    evaluate_xgb_for_zone,
    fit_all_horizons_for_zone,
    fit_xgb_for_all_zones,
    fit_xgb_for_zone,
    generate_xgb_residuals,
    predict_xgb,
)


def test_xgb_fit_predict_runs_and_aligns_index(synthetic_model_df) -> None:
    model = fit_xgb_for_zone(synthetic_model_df, 2, "zone_a")
    zone_test_df = synthetic_model_df.loc[
        (synthetic_model_df["zone"] == "zone_a")
        & (synthetic_model_df["split"] == "test")
    ]
    predictions = predict_xgb(model, zone_test_df)

    assert predictions.index.equals(zone_test_df.index)
    assert not predictions.isna().any()


def test_xgb_horizon_validation_works(synthetic_model_df) -> None:
    with pytest.raises(ValueError, match="Horizon must be one of"):
        fit_xgb_for_zone(synthetic_model_df, 8, "zone_a")


def test_xgb_results_table_has_expected_columns(synthetic_model_df) -> None:
    model, results = evaluate_xgb_for_zone(synthetic_model_df, 3, "zone_b")

    assert model.horizon == 3
    assert list(results.columns) == list(XGB_RESULTS_COLUMNS)
    assert results["zone"].tolist() == ["zone_b"]
    assert not results.isna().any().any()


def test_xgb_multi_horizon_and_multi_zone_helpers_work(synthetic_model_df) -> None:
    zone_models = fit_all_horizons_for_zone(synthetic_model_df, "zone_a")
    all_zone_models = fit_xgb_for_all_zones(synthetic_model_df, 1)
    _, combined_results = evaluate_xgb_for_all_zones(synthetic_model_df, 1)

    assert sorted(zone_models) == [1, 2, 3, 4, 5, 6, 7]
    assert sorted(all_zone_models) == ["zone_a", "zone_b"]
    assert combined_results["zone"].tolist() == ["zone_a", "zone_b"]


def test_xgb_residual_generation_returns_expected_shape(synthetic_model_df) -> None:
    model = fit_xgb_for_zone(synthetic_model_df, 1, "zone_a")
    zone_test_df = synthetic_model_df.loc[
        (synthetic_model_df["zone"] == "zone_a")
        & (synthetic_model_df["split"] == "test")
    ]
    residuals_df = generate_xgb_residuals(model, zone_test_df)

    assert residuals_df.columns.tolist() == [
        "date",
        "zone",
        "horizon",
        "actual",
        "prediction",
        "residual",
    ]
    assert len(residuals_df) == len(zone_test_df)
