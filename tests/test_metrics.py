# pylint: disable=missing-module-docstring, missing-function-docstring

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.evaluation.metrics import (
    RESULTS_TABLE_COLUMNS,
    build_results_table,
    mape,
    rmse,
    wmape,
)


def test_metrics_return_expected_values() -> None:
    actual = pd.Series([100.0, 200.0])
    predicted = pd.Series([110.0, 190.0])

    assert mape(actual, predicted) == pytest.approx(7.5)
    assert wmape(actual, predicted) == pytest.approx(6.6666666667)
    assert rmse(actual, predicted) == pytest.approx(10.0)


def test_metrics_validate_equal_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        rmse([1.0], [1.0, 2.0])


def test_mape_ignores_zero_actuals() -> None:
    assert mape([0.0, 100.0], [5.0, 90.0]) == pytest.approx(10.0)


def test_wmape_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="sum of absolute actuals is zero"):
        wmape([0.0, 0.0], [1.0, 2.0])


def test_build_results_table_returns_shared_schema() -> None:
    metadata = pd.DataFrame({"zone": ["zone_b", "zone_a"]})
    results = build_results_table(
        metadata,
        [100.0, 200.0],
        [110.0, 180.0],
        model_name="linear",
        horizon=3,
    )

    assert list(results.columns) == list(RESULTS_TABLE_COLUMNS)
    assert results["zone"].tolist() == ["zone_a", "zone_b"]
    assert results["horizon"].tolist() == [3, 3]
    assert math.isfinite(results.loc[0, "mape"])
