"""Canonical public configuration for the workshop runtime."""

from __future__ import annotations

import os
from pathlib import Path


ALLOWED_HORIZONS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
SUPPORTED_TFT_HORIZONS: tuple[int, ...] = ALLOWED_HORIZONS

REQUIRED_SCHEMA_FIELDS: tuple[str, ...] = (
    "date",
    "zone",
    "split",
    "demand",
    "temp",
    "hdd",
    "cdd",
    "dow",
    "is_weekend",
    "month",
    "lag1",
    "lag7",
    "target_h1",
    "target_h2",
    "target_h3",
    "target_h4",
    "target_h5",
    "target_h6",
    "target_h7",
)

NUMERIC_SCHEMA_FIELDS: tuple[str, ...] = (
    "demand",
    "temp",
    "hdd",
    "cdd",
    "dow",
    "is_weekend",
    "month",
    "lag1",
    "lag7",
    "target_h1",
    "target_h2",
    "target_h3",
    "target_h4",
    "target_h5",
    "target_h6",
    "target_h7",
)

GITHUB_REPO_OWNER = os.environ.get("SGA_WORKSHOP_GITHUB_REPO_OWNER", "RichardPovinelli")
GITHUB_REPO_NAME = os.environ.get("SGA_WORKSHOP_GITHUB_REPO_NAME", "SGA_workshop")

EXPECTED_DATASET_SCHEMA_VERSION = "1"
EXPECTED_DATASET_RUNTIME_COMPATIBILITY_VERSION = "1"
EXPECTED_TFT_CHECKPOINT_FORMAT_VERSION = "1"
EXPECTED_TFT_RUNTIME_COMPATIBILITY_VERSION = "1"

LOCAL_STATE_ROOT = Path(
    os.environ.get("SGA_WORKSHOP_LOCAL_STATE_ROOT", Path.home() / ".sga_workshop")
)
RELEASE_STAGING_ROOT = LOCAL_STATE_ROOT / "release_staging"

DATASET_RELEASE_TAG = os.environ.get("SGA_WORKSHOP_DATASET_RELEASE_TAG", "dataset-v1")
DATASET_BUILD_VERSION = os.environ.get("SGA_WORKSHOP_DATASET_BUILD_VERSION", "1")
DATASET_ASSET_FILENAME_TEMPLATE = "sga_workshop_dataset_v{version}.parquet.gz"
DATASET_ASSET_FILENAME = DATASET_ASSET_FILENAME_TEMPLATE.format(
    version=DATASET_BUILD_VERSION
)
DATASET_ARTIFACT_TYPE = "dataset"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
DATASET_LOCAL_DIR = LOCAL_STATE_ROOT / "data"
DATASET_LOCAL_PATH = DATASET_LOCAL_DIR / "sga_workshop_dataset.parquet.gz"
DATASET_MANIFEST_LOCAL_PATH = DATASET_LOCAL_DIR / DATASET_MANIFEST_FILENAME
DATASET_INSTALL_STATE_PATH = DATASET_LOCAL_DIR / "install_state.json"

TFT_RELEASE_TAG = os.environ.get("SGA_WORKSHOP_TFT_RELEASE_TAG", "tft-v1")
DEFAULT_TFT_BUNDLE_VERSION = os.environ.get("SGA_WORKSHOP_TFT_BUNDLE_VERSION", "1")
TFT_BUNDLE_FILENAME_TEMPLATE = "sga_workshop_tft_checkpoints_v{version}.tgz"
TFT_BUNDLE_FILENAME = TFT_BUNDLE_FILENAME_TEMPLATE.format(
    version=DEFAULT_TFT_BUNDLE_VERSION
)
TFT_ARTIFACT_TYPE = "tft_checkpoints"
TFT_CHECKPOINT_MAP_FILENAME = "checkpoint_map.json"
TFT_BUNDLE_URL_ENV_VAR = "SGA_WORKSHOP_TFT_BUNDLE_URL"
DATASET_URL_ENV_VAR = "SGA_WORKSHOP_DATASET_URL"

DEFAULT_TFT_ARTIFACT_ROOT = LOCAL_STATE_ROOT / "tft"
DEFAULT_TFT_BUNDLE_DIR = DEFAULT_TFT_ARTIFACT_ROOT / "bundles"
DEFAULT_TFT_INSTALL_ROOT = DEFAULT_TFT_ARTIFACT_ROOT / "current"
TFT_BUNDLE_LOCAL_PATH = DEFAULT_TFT_BUNDLE_DIR / TFT_BUNDLE_FILENAME
TFT_INSTALL_STATE_PATH = DEFAULT_TFT_ARTIFACT_ROOT / "install_state.json"

DEFAULT_DATASET_PATH = DATASET_LOCAL_PATH
