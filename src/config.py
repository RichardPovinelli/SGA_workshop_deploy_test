"""Canonical public configuration for the workshop runtime."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"


ALLOWED_HORIZONS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
SUPPORTED_MLP_FORECAST_HORIZONS: tuple[int, ...] = ALLOWED_HORIZONS

REQUIRED_SCHEMA_FIELDS: tuple[str, ...] = (
    "date",
    "split",
    "demand",
    "temp",
    "hdd",
    "cdd",
    "dow",
    "is_weekend",
    "month",
    "demand_lag1",
    "demand_lag2",
    "demand_lag3",
    "demand_lag4",
    "demand_lag5",
    "demand_lag6",
    "demand_lag7",
    "temp_lag1",
    "temp_lag2",
    "hdd_lag1",
    "hdd_lag2",
    "cdd_lag1",
    "cdd_lag2",
    "target_h1",
    "target_h2",
    "target_h3",
    "target_h4",
    "target_h5",
    "target_h6",
    "target_h7",
    "temp_h1",
    "temp_h2",
    "temp_h3",
    "temp_h4",
    "temp_h5",
    "temp_h6",
    "temp_h7",
    "hdd_h1",
    "hdd_h2",
    "hdd_h3",
    "hdd_h4",
    "hdd_h5",
    "hdd_h6",
    "hdd_h7",
    "cdd_h1",
    "cdd_h2",
    "cdd_h3",
    "cdd_h4",
    "cdd_h5",
    "cdd_h6",
    "cdd_h7",
)

NUMERIC_SCHEMA_FIELDS: tuple[str, ...] = (
    "demand",
    "temp",
    "hdd",
    "cdd",
    "dow",
    "is_weekend",
    "month",
    "demand_lag1",
    "demand_lag2",
    "demand_lag3",
    "demand_lag4",
    "demand_lag5",
    "demand_lag6",
    "demand_lag7",
    "temp_lag1",
    "temp_lag2",
    "hdd_lag1",
    "hdd_lag2",
    "cdd_lag1",
    "cdd_lag2",
    "target_h1",
    "target_h2",
    "target_h3",
    "target_h4",
    "target_h5",
    "target_h6",
    "target_h7",
    "temp_h1",
    "temp_h2",
    "temp_h3",
    "temp_h4",
    "temp_h5",
    "temp_h6",
    "temp_h7",
    "hdd_h1",
    "hdd_h2",
    "hdd_h3",
    "hdd_h4",
    "hdd_h5",
    "hdd_h6",
    "hdd_h7",
    "cdd_h1",
    "cdd_h2",
    "cdd_h3",
    "cdd_h4",
    "cdd_h5",
    "cdd_h6",
    "cdd_h7",
)

GITHUB_REPO_OWNER = os.environ.get("SGA_WORKSHOP_GITHUB_REPO_OWNER", "RichardPovinelli")
GITHUB_REPO_NAME = os.environ.get("SGA_WORKSHOP_GITHUB_REPO_NAME", "SGA_workshop")

EXPECTED_DATASET_SCHEMA_VERSION = "1"
EXPECTED_DATASET_RUNTIME_COMPATIBILITY_VERSION = "1"
EXPECTED_MLP_FORECAST_CHECKPOINT_FORMAT_VERSION = "1"
EXPECTED_MLP_FORECAST_RUNTIME_COMPATIBILITY_VERSION = "1"

LOCAL_STATE_ROOT = Path(os.environ.get("SGA_WORKSHOP_LOCAL_STATE_ROOT", Path.home() / ".sga_workshop"))
RELEASE_STAGING_ROOT = LOCAL_STATE_ROOT / "release_staging"

DATASET_RELEASE_TAG = os.environ.get("SGA_WORKSHOP_DATASET_RELEASE_TAG", "dataset-v1")
DATASET_BUILD_VERSION = os.environ.get("SGA_WORKSHOP_DATASET_BUILD_VERSION", "1")
DATASET_ASSET_FILENAME_TEMPLATE = "sga_workshop_dataset_v{version}.parquet.gz"
DATASET_ASSET_FILENAME = DATASET_ASSET_FILENAME_TEMPLATE.format(version=DATASET_BUILD_VERSION)
DATASET_ARTIFACT_TYPE = "dataset"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
DATASET_LOCAL_DIR = LOCAL_STATE_ROOT / "data"
DATASET_LOCAL_PATH = DATASET_LOCAL_DIR / "sga_workshop_dataset.parquet.gz"
DATASET_MANIFEST_LOCAL_PATH = DATASET_LOCAL_DIR / DATASET_MANIFEST_FILENAME
DATASET_INSTALL_STATE_PATH = DATASET_LOCAL_DIR / "install_state.json"

MLP_FORECAST_RELEASE_TAG = os.environ.get("SGA_WORKSHOP_MLP_FORECAST_RELEASE_TAG", "mlp_forecast-v1")
DEFAULT_MLP_FORECAST_BUNDLE_VERSION = os.environ.get("SGA_WORKSHOP_MLP_FORECAST_BUNDLE_VERSION", "1")
MLP_FORECAST_BUNDLE_FILENAME_TEMPLATE = "sga_workshop_mlp_forecast_checkpoints_v{version}.tgz"
MLP_FORECAST_BUNDLE_FILENAME = MLP_FORECAST_BUNDLE_FILENAME_TEMPLATE.format(version=DEFAULT_MLP_FORECAST_BUNDLE_VERSION)
MLP_FORECAST_ARTIFACT_TYPE = "mlp_forecast_checkpoints"
MLP_FORECAST_CHECKPOINT_MAP_FILENAME = "checkpoint_map.json"
MLP_FORECAST_BUNDLE_URL_ENV_VAR = "SGA_WORKSHOP_MLP_FORECAST_BUNDLE_URL"
DATASET_URL_ENV_VAR = "SGA_WORKSHOP_DATASET_URL"

DEFAULT_MLP_FORECAST_ARTIFACT_ROOT = LOCAL_STATE_ROOT / "mlp_forecast"
DEFAULT_MLP_FORECAST_BUNDLE_DIR = DEFAULT_MLP_FORECAST_ARTIFACT_ROOT / "bundles"
DEFAULT_MLP_FORECAST_INSTALL_ROOT = ARTIFACTS_ROOT / "mlp_forecast"
MLP_FORECAST_BUNDLE_LOCAL_PATH = DEFAULT_MLP_FORECAST_BUNDLE_DIR / MLP_FORECAST_BUNDLE_FILENAME
MLP_FORECAST_INSTALL_STATE_PATH = DEFAULT_MLP_FORECAST_ARTIFACT_ROOT / "install_state.json"

TFT_ARTIFACT_TYPE = "tft_checkpoints"
EXPECTED_TFT_CHECKPOINT_FORMAT_VERSION = "1"
EXPECTED_TFT_RUNTIME_COMPATIBILITY_VERSION = "1"
DEFAULT_TFT_ARTIFACT_ROOT = ARTIFACTS_ROOT / "tft"
TFT_CHECKPOINT_FILENAME = "tft_model.ckpt"
TFT_CHECKPOINT_MAP_FILENAME = "checkpoint_map.json"
TFT_MAX_ENCODER_LENGTH = 30
TFT_MAX_PREDICTION_LENGTH = 7

DATASET_PATH = ARTIFACTS_ROOT / "sga_workshop_dataset.parquet.gz"
DATASET_MANIFEST_PATH = ARTIFACTS_ROOT / "dataset_manifest.json"
DEFAULT_DATASET_PATH = DATASET_PATH
