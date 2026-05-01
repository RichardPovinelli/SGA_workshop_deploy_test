# pylint: disable=missing-module-docstring, missing-function-docstring

from __future__ import annotations

import pandas as pd
import pytest

from src.config import (
    EXPECTED_DATASET_RUNTIME_COMPATIBILITY_VERSION,
    EXPECTED_DATASET_SCHEMA_VERSION,
    EXPECTED_TFT_CHECKPOINT_FORMAT_VERSION,
    EXPECTED_TFT_RUNTIME_COMPATIBILITY_VERSION,
)
from src.data.schema import get_sorted_zones, validate_dataset_schema


def make_valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "zone": ["zone_b", "zone_a", "zone_b"],
            "split": ["train", "train", "test"],
            "demand": [100.0, 101.0, 99.0],
            "temp": [20.0, 22.0, 19.5],
            "hdd": [5.0, 4.0, 6.0],
            "cdd": [0.0, 0.0, 0.0],
            "dow": [1, 2, 3],
            "is_weekend": [0, 0, 0],
            "month": [1, 1, 1],
            "lag1": [98.0, 100.0, 101.0],
            "lag7": [97.0, 98.0, 99.0],
            "target_h1": [101.0, 99.0, 100.0],
            "target_h2": [102.0, 100.0, 101.0],
            "target_h3": [103.0, 101.0, 102.0],
            "target_h4": [104.0, 102.0, 103.0],
            "target_h5": [105.0, 103.0, 104.0],
            "target_h6": [106.0, 104.0, 105.0],
            "target_h7": [107.0, 105.0, 106.0],
        }
    )


def test_validate_dataset_schema_accepts_valid_fixture() -> None:
    df = make_valid_df()

    validate_dataset_schema(df)


def test_validate_dataset_schema_rejects_missing_column() -> None:
    df = make_valid_df().drop(columns=["target_h7"])

    with pytest.raises(ValueError, match="missing required fields"):
        validate_dataset_schema(df)


def test_validate_dataset_schema_rejects_bad_split_values() -> None:
    df = make_valid_df().copy()
    df.loc[2, "split"] = "validation"

    with pytest.raises(ValueError, match="exactly"):
        validate_dataset_schema(df)


def test_sorted_zones_are_deterministic() -> None:
    df = make_valid_df()

    assert get_sorted_zones(df) == ["zone_a", "zone_b"]


def test_public_config_exposes_compatibility_versions() -> None:
    assert EXPECTED_DATASET_SCHEMA_VERSION == "1"
    assert EXPECTED_DATASET_RUNTIME_COMPATIBILITY_VERSION == "1"
    assert EXPECTED_TFT_CHECKPOINT_FORMAT_VERSION == "1"
    assert EXPECTED_TFT_RUNTIME_COMPATIBILITY_VERSION == "1"
