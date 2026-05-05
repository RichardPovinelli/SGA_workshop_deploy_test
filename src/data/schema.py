"""Dataset schema validation helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import NUMERIC_SCHEMA_FIELDS, REQUIRED_SCHEMA_FIELDS


MODELING_REQUIRED_FIELDS: tuple[str, ...] = REQUIRED_SCHEMA_FIELDS
ALLOWED_SPLITS: frozenset[str] = frozenset({"train", "test"})


def _missing_fields(columns: Iterable[str]) -> list[str]:
    present = set(columns)
    return [field for field in REQUIRED_SCHEMA_FIELDS if field not in present]


def validate_dataset_schema(df: pd.DataFrame) -> None:
    """Validate the public dataset contract."""
    missing_fields = _missing_fields(df.columns)
    if missing_fields:
        raise ValueError(f"Dataset is missing required fields: {missing_fields}")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("Column 'date' must be parsed as datetime.")

    split_values = set(df["split"].dropna().unique().tolist())
    if split_values != ALLOWED_SPLITS:
        unexpected_values = sorted(split_values.symmetric_difference(ALLOWED_SPLITS))
        raise ValueError(
            "Column 'split' must contain exactly {'train', 'test'} "
            f"but found: {sorted(split_values)}; mismatch: {unexpected_values}"
        )

    for field in NUMERIC_SCHEMA_FIELDS:
        if not is_numeric_dtype(df[field]):
            raise ValueError(f"Column '{field}' must be numeric.")

    missing_modeling_values = cast(
        pd.Series,
        df.loc[:, MODELING_REQUIRED_FIELDS].isna().sum(axis=0),
    )
    invalid_fields = missing_modeling_values[missing_modeling_values > 0].index.tolist()
    if invalid_fields:
        raise ValueError(f"Required modeling columns must not contain missing values: {invalid_fields}")
