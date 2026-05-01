# pylint: disable=missing-module-docstring, missing-function-docstring

from __future__ import annotations

import pytest

from src.evaluation.dm_test import diebold_mariano_test


def test_dm_wrapper_returns_expected_keys() -> None:
    result = diebold_mariano_test(
        actuals=[10.0, 11.0, 12.0, 13.0],
        predictions_a=[9.5, 10.5, 11.5, 12.5],
        predictions_b=[9.0, 10.0, 11.0, 12.0],
        loss="absolute",
    )

    assert set(result) == {"statistic", "p_value", "n_obs", "loss"}
    assert result["n_obs"] == 4
    assert result["loss"] == "absolute"
    assert 0.0 <= float(result["p_value"]) <= 1.0


def test_dm_wrapper_rejects_invalid_loss() -> None:
    with pytest.raises(ValueError, match="loss must be one of"):
        diebold_mariano_test([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], loss="foo")
