"""Shared external-asset contracts for runtime, bootstrap, and rehearsal."""

from __future__ import annotations

import json
import os
import platform
import shutil
import tarfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DATASET_ARTIFACT_TYPE,
    DATASET_ASSET_FILENAME,
    DATASET_INSTALL_STATE_PATH,
    DATASET_LOCAL_PATH,
    DATASET_MANIFEST_FILENAME,
    DATASET_MANIFEST_LOCAL_PATH,
    DATASET_RELEASE_TAG,
    EXPECTED_DATASET_RUNTIME_COMPATIBILITY_VERSION,
    EXPECTED_DATASET_SCHEMA_VERSION,
    EXPECTED_TFT_RUNTIME_COMPATIBILITY_VERSION,
    GITHUB_REPO_NAME,
    GITHUB_REPO_OWNER,
    RELEASE_STAGING_ROOT,
    TFT_ARTIFACT_TYPE,
    TFT_BUNDLE_FILENAME,
    TFT_BUNDLE_LOCAL_PATH,
    TFT_INSTALL_STATE_PATH,
    TFT_RELEASE_TAG,
)
from src.data.schema import get_sorted_zones, validate_dataset_schema
from src.models.tft_inference import load_installed_tft_checkpoint_map


RELEASE_ASSETS_MANIFEST_FILENAME = "release_assets_manifest.json"
SUPPORTED_POSIX_SYSTEMS: tuple[str, ...] = ("Linux", "Darwin")
LOCAL_STATE_ROOT_ENV_VAR = "SGA_WORKSHOP_LOCAL_STATE_ROOT"


class DatasetCompatibilityError(ValueError):
    """Raised when installed dataset compatibility metadata is invalid."""


class InstallStateError(ValueError):
    """Raised when install-state metadata is missing or inconsistent."""


@dataclass(frozen=True)
class AssetDescriptor:
    """Stable descriptor for one externally published workshop asset."""

    artifact_type: str
    release_tag: str
    filename: str
    local_install_path: str
    compatibility_version: str


def build_github_release_asset_url(
    owner: str, repo: str, tag: str, filename: str
) -> str:
    """Return the canonical GitHub release asset URL."""
    return f"https://github.com/{owner}/{repo}/releases/download/{tag}/{filename}"


def build_dataset_asset_filename(version: str) -> str:
    """Return the canonical versioned dataset artifact filename."""
    if not version or version.strip() == "":
        raise ValueError("version must be a non-empty string.")
    return f"sga_workshop_dataset_v{version}.parquet.gz"


def get_dataset_asset_descriptor() -> AssetDescriptor:
    """Return the canonical dataset asset descriptor."""
    return AssetDescriptor(
        artifact_type=DATASET_ARTIFACT_TYPE,
        release_tag=DATASET_RELEASE_TAG,
        filename=DATASET_ASSET_FILENAME,
        local_install_path=str(DATASET_LOCAL_PATH),
        compatibility_version=EXPECTED_DATASET_RUNTIME_COMPATIBILITY_VERSION,
    )


def get_tft_asset_descriptor() -> AssetDescriptor:
    """Return the canonical TFT asset descriptor."""
    return AssetDescriptor(
        artifact_type=TFT_ARTIFACT_TYPE,
        release_tag=TFT_RELEASE_TAG,
        filename=TFT_BUNDLE_FILENAME,
        local_install_path=str(TFT_BUNDLE_LOCAL_PATH.parent),
        compatibility_version=EXPECTED_TFT_RUNTIME_COMPATIBILITY_VERSION,
    )


def get_dataset_asset_url() -> str:
    """Return the canonical dataset release URL."""
    descriptor = get_dataset_asset_descriptor()
    return build_github_release_asset_url(
        GITHUB_REPO_OWNER,
        GITHUB_REPO_NAME,
        descriptor.release_tag,
        descriptor.filename,
    )


def get_tft_asset_url() -> str:
    """Return the canonical TFT release URL."""
    descriptor = get_tft_asset_descriptor()
    return build_github_release_asset_url(
        GITHUB_REPO_OWNER,
        GITHUB_REPO_NAME,
        descriptor.release_tag,
        descriptor.filename,
    )


def dataset_manifest_path_for(dataset_path: str | Path) -> Path:
    """Return the stable local dataset manifest path for an installed artifact."""
    path = Path(dataset_path)
    if path == DATASET_LOCAL_PATH:
        return DATASET_MANIFEST_LOCAL_PATH
    return path.parent / DATASET_MANIFEST_FILENAME


def _local_state_root() -> Path:
    return Path(
        os.environ.get(
            LOCAL_STATE_ROOT_ENV_VAR,
            str(DATASET_LOCAL_PATH.parent.parent),
        )
    )


def _dataset_local_dir() -> Path:
    return _local_state_root() / "data"


def _dataset_local_path() -> Path:
    return _dataset_local_dir() / DATASET_LOCAL_PATH.name


def _dataset_manifest_local_path() -> Path:
    return _dataset_local_dir() / DATASET_MANIFEST_FILENAME


def _dataset_install_state_path() -> Path:
    return _dataset_local_dir() / DATASET_INSTALL_STATE_PATH.name


def _tft_artifact_root() -> Path:
    return _local_state_root() / "tft"


def _tft_bundle_local_path() -> Path:
    return _tft_artifact_root() / "bundles" / TFT_BUNDLE_FILENAME


def _tft_install_state_path() -> Path:
    return _tft_artifact_root() / TFT_INSTALL_STATE_PATH.name


def _resolve_install_state_path(state_path: str | Path) -> Path:
    path = Path(state_path)
    if LOCAL_STATE_ROOT_ENV_VAR not in os.environ:
        return path
    if path == DATASET_INSTALL_STATE_PATH or (
        path.name == DATASET_INSTALL_STATE_PATH.name and path.parent.name == "data"
    ):
        return _dataset_install_state_path()
    if path == TFT_INSTALL_STATE_PATH or (
        path.name == TFT_INSTALL_STATE_PATH.name and path.parent.name == "tft"
    ):
        return _tft_install_state_path()
    return path


def ensure_supported_bootstrap_platform(system_name: str | None = None) -> str:
    """Validate that bootstrap is running on a supported POSIX-style platform."""
    resolved_system_name = system_name or platform.system()
    if resolved_system_name not in SUPPORTED_POSIX_SYSTEMS:
        raise RuntimeError(
            "bootstrap.sh supports POSIX-style environments only. "
            "Use Google Colab, a POSIX shell, or WSL on Windows."
        )
    return resolved_system_name


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def validate_dataset_manifest_payload(payload: dict[str, Any]) -> None:
    """Validate the installed dataset manifest contract."""
    required_fields = (
        "artifact_type",
        "schema_version",
        "runtime_compatibility_version",
        "build_version",
        "rows",
        "columns",
        "zone_list",
        "min_date",
        "max_date",
        "split_counts",
        "build_seed",
        "build_timestamp",
        "artifact_filename",
    )
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        raise DatasetCompatibilityError(
            f"Dataset manifest is missing required field(s): {missing_fields}."
        )
    if payload["artifact_type"] != DATASET_ARTIFACT_TYPE:
        raise DatasetCompatibilityError(
            "Dataset manifest artifact_type must be 'dataset'."
        )
    if str(payload["schema_version"]) != EXPECTED_DATASET_SCHEMA_VERSION:
        raise DatasetCompatibilityError(
            "Dataset schema_version is incompatible with the current runtime."
        )
    if (
        str(payload["runtime_compatibility_version"])
        != EXPECTED_DATASET_RUNTIME_COMPATIBILITY_VERSION
    ):
        raise DatasetCompatibilityError(
            "Dataset runtime_compatibility_version is incompatible with the current runtime."
        )


def validate_installed_dataset(
    dataset_path: str | Path = DATASET_LOCAL_PATH,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load and validate an installed dataset plus its compatibility manifest."""
    resolved_dataset_path = Path(dataset_path)
    resolved_manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else dataset_manifest_path_for(resolved_dataset_path)
    )
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(
            f"Installed dataset artifact is missing: {resolved_dataset_path}"
        )
    if not resolved_manifest_path.exists():
        raise DatasetCompatibilityError(
            f"Dataset manifest is missing: {resolved_manifest_path}"
        )

    manifest = _read_json(resolved_manifest_path)
    validate_dataset_manifest_payload(manifest)

    df = pd.read_parquet(resolved_dataset_path, engine="pyarrow")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    validate_dataset_schema(df)

    if int(manifest["rows"]) != len(df):
        raise DatasetCompatibilityError(
            "Dataset manifest row count does not match installed data."
        )
    if list(manifest["columns"]) != list(df.columns):
        raise DatasetCompatibilityError(
            "Dataset manifest columns do not match installed data."
        )
    if list(manifest["zone_list"]) != get_sorted_zones(df):
        raise DatasetCompatibilityError(
            "Dataset manifest zone ordering does not match installed data."
        )

    return df


def load_install_state(state_path: str | Path) -> dict[str, Any] | None:
    """Load local-only install-state metadata if it exists."""
    path = _resolve_install_state_path(state_path)
    if not path.exists():
        return None
    payload = _read_json(path)
    required_fields = (
        "artifact_type",
        "installed_tag",
        "installed_filename",
        "install_timestamp",
        "compatibility_version",
    )
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        raise InstallStateError(
            f"Install-state metadata is missing required field(s): {missing_fields}."
        )
    return payload


def write_install_state(state_path: str | Path, payload: dict[str, Any]) -> None:
    """Write local-only install-state metadata."""
    _write_json(Path(state_path), payload)


def install_dataset_from_file(
    source_dataset_path: str | Path,
    *,
    source_manifest_path: str | Path | None = None,
    requested_release_tag: str | None = None,
    requested_filename: str | None = None,
    install_timestamp: str | None = None,
) -> Path:
    """Install a dataset artifact plus manifest into the canonical local state path."""
    source_path = Path(source_dataset_path)
    source_manifest = (
        Path(source_manifest_path)
        if source_manifest_path is not None
        else dataset_manifest_path_for(source_path)
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset artifact is missing: {source_path}")
    if not source_manifest.exists():
        raise FileNotFoundError(
            f"Source dataset manifest is missing: {source_manifest}"
        )

    manifest = _read_json(source_manifest)
    validate_dataset_manifest_payload(manifest)
    if (
        requested_release_tag is not None
        and manifest.get("release_tag") != requested_release_tag
    ):
        raise DatasetCompatibilityError(
            "Requested dataset release_tag does not match source manifest."
        )
    if requested_filename is not None and source_path.name != requested_filename:
        raise DatasetCompatibilityError(
            "Requested dataset filename does not match source artifact."
        )

    dataset_local_path = _dataset_local_path()
    dataset_manifest_local_path = _dataset_manifest_local_path()
    dataset_install_state_path = _dataset_install_state_path()

    dataset_local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dataset_local_path)
    shutil.copy2(source_manifest, dataset_manifest_local_path)
    write_install_state(
        dataset_install_state_path,
        {
            "artifact_type": DATASET_ARTIFACT_TYPE,
            "installed_tag": manifest.get(
                "release_tag",
                requested_release_tag or DATASET_RELEASE_TAG,
            ),
            "installed_filename": source_path.name,
            "install_timestamp": install_timestamp
            or datetime.now(timezone.utc).isoformat(),
            "compatibility_version": manifest["runtime_compatibility_version"],
            "local_install_path": str(dataset_local_path),
        },
    )
    return dataset_local_path


def install_tft_bundle_from_file(
    source_bundle_path: str | Path,
    *,
    requested_release_tag: str | None = None,
    requested_filename: str | None = None,
    install_timestamp: str | None = None,
) -> Path:
    """Install and validate a TFT bundle into the canonical local state path."""
    source_path = Path(source_bundle_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source TFT bundle is missing: {source_path}")
    if requested_filename is not None and source_path.name != requested_filename:
        raise ValueError(
            "Requested TFT bundle filename does not match source artifact."
        )

    tft_bundle_local_path = _tft_bundle_local_path()
    tft_install_state_path = _tft_install_state_path()

    tft_bundle_local_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, tft_bundle_local_path)
    if _dataset_local_path().parent == tft_bundle_local_path.parent:
        raise ValueError("Dataset and TFT install roots must remain separate.")

    if Path(tft_bundle_local_path.parent / "..").resolve() == Path(".").resolve():
        raise ValueError("TFT local state must live outside the repo root.")

    if (
        Path(tft_bundle_local_path.parent.parent).resolve()
        != Path(tft_install_state_path.parent).resolve()
    ):
        tft_install_state_path.parent.mkdir(parents=True, exist_ok=True)
    if Path(tft_bundle_local_path.parent).exists():
        pass

    if Path(tft_bundle_local_path.parent.parent).exists():
        pass

    extract_root = Path(tft_bundle_local_path.parent.parent / "current")
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tft_bundle_local_path, "r:gz") as archive:
        archive.extractall(extract_root)

    installed_map = load_installed_tft_checkpoint_map(
        artifact_root=extract_root,
        bundle_path=tft_bundle_local_path,
    )
    write_install_state(
        tft_install_state_path,
        {
            "artifact_type": TFT_ARTIFACT_TYPE,
            "installed_tag": requested_release_tag or TFT_RELEASE_TAG,
            "installed_filename": source_path.name,
            "install_timestamp": install_timestamp
            or datetime.now(timezone.utc).isoformat(),
            "compatibility_version": installed_map.runtime_compatibility_version,
            "extracted_root": str(installed_map.artifact_root),
        },
    )
    return extract_root


def build_release_assets_manifest() -> dict[str, Any]:
    """Return the canonical staged release-assets manifest payload."""
    dataset_descriptor = get_dataset_asset_descriptor()
    tft_descriptor = get_tft_asset_descriptor()
    return {
        "dataset": {
            **asdict(dataset_descriptor),
            "artifact_type": DATASET_ARTIFACT_TYPE,
        },
        "tft_checkpoints": {
            **asdict(tft_descriptor),
            "artifact_type": TFT_ARTIFACT_TYPE,
        },
    }


def write_release_assets_manifest(
    staging_root: str | Path = RELEASE_STAGING_ROOT,
) -> Path:
    """Write the staged release-assets manifest into local-only staging."""
    root = Path(staging_root)
    payload = build_release_assets_manifest()
    manifest_path = root / RELEASE_ASSETS_MANIFEST_FILENAME
    _write_json(manifest_path, payload)
    return manifest_path
