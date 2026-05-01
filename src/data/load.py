"""Data loading helpers for the public runtime."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATASET_MANIFEST_LOCAL_PATH, DEFAULT_DATASET_PATH
from src.runtime_assets import validate_installed_dataset


def load_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    manifest_path: str | Path | None = DATASET_MANIFEST_LOCAL_PATH,
) -> pd.DataFrame:
    """Load the derived dataset and validate its installed compatibility metadata."""
    path = Path(dataset_path)
    suffixes = path.suffixes
    if suffixes not in [[".parquet"], [".parquet", ".gz"]]:
        raise ValueError(f"Dataset path must end with .parquet or .parquet.gz: {path}")
    return validate_installed_dataset(path, manifest_path)
