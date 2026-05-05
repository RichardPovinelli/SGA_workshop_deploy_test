"""Split and feature-selection helpers."""

from __future__ import annotations

import pandas as pd

from src.config import ALLOWED_HORIZONS, REQUIRED_SCHEMA_FIELDS


METADATA_FIELDS: tuple[str, ...] = ("date", "split")
TARGET_FIELDS: tuple[str, ...] = tuple(field for field in REQUIRED_SCHEMA_FIELDS if field.startswith("target_h"))


def validate_horizon(horizon: int) -> int:
    """Validate a direct-forecast horizon."""
    if horizon not in ALLOWED_HORIZONS:
        raise ValueError(f"Horizon must be one of {ALLOWED_HORIZONS}, got {horizon}.")
    return horizon


def get_train_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows marked for training."""
    return df.loc[df["split"] == "train"].copy()


def get_test_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows marked for testing."""
    return df.loc[df["split"] == "test"].copy()


def get_feature_columns(horizon: int | None = None) -> list[str]:
    """Return feature columns for a given forecast horizon.

    When horizon is provided, includes perfect-weather columns (temp_hN, hdd_hN,
    cdd_hN) for that horizon only. When omitted, returns base features only.
    """
    perfect_weather_fields = frozenset(f"{var}_h{h}" for h in ALLOWED_HORIZONS for var in ("temp", "hdd", "cdd"))
    base_cols = [
        field
        for field in REQUIRED_SCHEMA_FIELDS
        if field not in METADATA_FIELDS and field not in TARGET_FIELDS and field not in perfect_weather_fields
    ]
    if horizon is None:
        return base_cols
    validated = validate_horizon(horizon)
    return base_cols + [f"temp_h{validated}", f"hdd_h{validated}", f"cdd_h{validated}"]


def get_target_column(horizon: int) -> str:
    """Return the target column name for a direct-forecast horizon."""
    validated_horizon = validate_horizon(horizon)
    return f"target_h{validated_horizon}"


__all__ = [
    "get_feature_columns",
    "get_sorted_zones",
    "get_target_column",
    "get_test_df",
    "get_train_df",
    "validate_horizon",
]
