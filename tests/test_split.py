# pylint: disable=missing-module-docstring, missing-function-docstring

from __future__ import annotations

import pytest

from tests.test_schema import make_valid_df
from src.data.split import (
    get_feature_columns,
    get_target_column,
    get_test_df,
    get_train_df,
    validate_horizon,
)


def test_validate_horizon_accepts_supported_values() -> None:
    assert validate_horizon(1) == 1
    assert validate_horizon(7) == 7


def test_validate_horizon_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Horizon must be one of"):
        validate_horizon(8)


def test_get_feature_columns_excludes_metadata_and_targets() -> None:
    feature_columns = get_feature_columns()

    assert "date" not in feature_columns
    assert "zone" not in feature_columns
    assert "split" not in feature_columns
    assert "target_h1" not in feature_columns
    assert "target_h7" not in feature_columns
    assert "demand" in feature_columns
    assert "lag7" in feature_columns


def test_get_target_column_returns_expected_name() -> None:
    assert get_target_column(3) == "target_h3"


def test_split_helpers_return_expected_rows() -> None:
    df = make_valid_df()

    train_df = get_train_df(df)
    test_df = get_test_df(df)

    assert train_df["split"].tolist() == ["train", "train"]
    assert test_df["split"].tolist() == ["test"]
