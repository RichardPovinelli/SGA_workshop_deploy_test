"""Inference-only TFT checkpoint loading and prediction helpers."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.config import (
    ALLOWED_HORIZONS,
    DEFAULT_TFT_BUNDLE_DIR,
    DEFAULT_TFT_BUNDLE_VERSION,
    DEFAULT_TFT_INSTALL_ROOT,
    EXPECTED_TFT_CHECKPOINT_FORMAT_VERSION,
    EXPECTED_TFT_RUNTIME_COMPATIBILITY_VERSION,
    SUPPORTED_TFT_HORIZONS,
    TFT_ARTIFACT_TYPE,
    TFT_BUNDLE_FILENAME_TEMPLATE,
    TFT_BUNDLE_URL_ENV_VAR,
    TFT_CHECKPOINT_MAP_FILENAME,
)
from src.data.split import get_target_column, validate_horizon
from src.evaluation.metrics import build_results_table


class TFTDependencyError(ImportError):
    """Raised when the runtime TFT dependency is not available."""


class TFTBundleNotFoundError(FileNotFoundError):
    """Raised when the bundled TFT checkpoint asset is missing."""


class TFTExtractionError(FileNotFoundError):
    """Raised when extracted TFT files are missing or incomplete."""


class TFTCheckpointMapError(ValueError):
    """Raised when the checkpoint map is malformed."""


class TFTCompatibilityError(ValueError):
    """Raised when installed TFT artifact metadata is incompatible."""


@dataclass(frozen=True)
class InstalledTFTCheckpointMap:  # pylint: disable=too-many-instance-attributes
    """Resolved, validated checkpoint map for horizons 1 through 7."""

    artifact_root: Path
    bundle_path: Path
    manifest_path: Path
    checkpoints: dict[int, Path]
    artifact_type: str
    build_version: str
    checkpoint_format_version: str
    runtime_compatibility_version: str
    bundle_url: str | None


def build_tft_bundle_filename(version: str) -> str:
    """Return the canonical TFT bundle filename for a release version."""
    if not version or version.strip() == "":
        raise ValueError("version must be a non-empty string.")
    return TFT_BUNDLE_FILENAME_TEMPLATE.format(version=version)


def _resolve_tft_bundle_path(
    *,
    bundle_path: str | Path | None = None,
    bundle_version: str | None = None,
) -> Path:
    if bundle_path is not None:
        return Path(bundle_path)
    resolved_version = bundle_version or DEFAULT_TFT_BUNDLE_VERSION
    return DEFAULT_TFT_BUNDLE_DIR / build_tft_bundle_filename(resolved_version)


def _resolve_tft_artifact_root(artifact_root: str | Path | None = None) -> Path:
    return (
        Path(artifact_root) if artifact_root is not None else DEFAULT_TFT_INSTALL_ROOT
    )


def _import_tft_dependency(module_name: str = "torch") -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise TFTDependencyError(
            "TFT inference dependencies are missing. Install the required TFT runtime "
            f"package(s) for the workshop environment before loading checkpoints: {module_name!r}."
        ) from exc


def _validate_checkpoint_map_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TFTCheckpointMapError("TFT checkpoint map must be a JSON object.")

    required_top_level_fields = (
        "artifact_type",
        "build_version",
        "checkpoint_format_version",
        "runtime_compatibility_version",
        "supported_horizons",
        "horizons",
    )
    missing_fields = [
        field for field in required_top_level_fields if field not in payload
    ]
    if missing_fields:
        raise TFTCheckpointMapError(
            f"TFT checkpoint map is missing required field(s): {missing_fields}."
        )

    horizons = payload["horizons"]
    if not isinstance(horizons, dict):
        raise TFTCheckpointMapError(
            "TFT checkpoint map field 'horizons' must be an object."
        )
    if payload["artifact_type"] != TFT_ARTIFACT_TYPE:
        raise TFTCheckpointMapError(
            "TFT checkpoint map artifact_type must be 'tft_checkpoints'."
        )

    expected_keys = {str(horizon) for horizon in ALLOWED_HORIZONS}
    observed_keys = set(horizons.keys())
    if observed_keys != expected_keys:
        raise TFTCheckpointMapError(
            "TFT checkpoint map must define horizons 1 through 7 exactly; "
            f"found keys: {sorted(observed_keys)}."
        )
    supported_horizons = tuple(payload["supported_horizons"])
    if tuple(int(horizon) for horizon in supported_horizons) != SUPPORTED_TFT_HORIZONS:
        raise TFTCheckpointMapError(
            "TFT checkpoint map supported_horizons must match 1 through 7 exactly."
        )


def _resolve_relative_checkpoint_path(artifact_root: Path, relative_path: str) -> Path:
    checkpoint_path = Path(relative_path)
    if checkpoint_path.is_absolute():
        raise TFTCheckpointMapError(
            "Checkpoint paths must be relative to the extracted TFT root."
        )
    if not relative_path or relative_path.strip() == "":
        raise TFTCheckpointMapError(
            "Checkpoint paths must be non-empty relative paths."
        )
    if ".." in checkpoint_path.parts:
        raise TFTCheckpointMapError(
            "Checkpoint paths must stay within the extracted TFT root; '..' is not allowed."
        )
    return artifact_root / checkpoint_path


def load_installed_tft_checkpoint_map(  # pylint: disable=too-many-locals
    *,
    artifact_root: str | Path | None = None,
    bundle_path: str | Path | None = None,
    bundle_version: str | None = None,
) -> InstalledTFTCheckpointMap:
    """Load and validate the installed TFT checkpoint layout."""
    resolved_artifact_root = _resolve_tft_artifact_root(artifact_root)
    resolved_bundle_path = _resolve_tft_bundle_path(
        bundle_path=bundle_path,
        bundle_version=bundle_version,
    )

    if not resolved_artifact_root.exists():
        if not resolved_bundle_path.exists():
            raise TFTBundleNotFoundError(
                "TFT checkpoint bundle is missing. Expected bundle asset at "
                f"{resolved_bundle_path}."
            )
        raise TFTExtractionError(
            "TFT checkpoint bundle was found but the extracted checkpoint root is missing: "
            f"{resolved_artifact_root}."
        )

    manifest_path = resolved_artifact_root / TFT_CHECKPOINT_MAP_FILENAME
    if not manifest_path.exists():
        raise TFTExtractionError(
            "Extracted TFT checkpoint root is missing the required manifest "
            f"{TFT_CHECKPOINT_MAP_FILENAME}: {manifest_path}."
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TFTCheckpointMapError(
            f"TFT checkpoint map is not valid JSON: {manifest_path}."
        ) from exc

    _validate_checkpoint_map_payload(payload)

    checkpoint_format_version = str(payload["checkpoint_format_version"])
    runtime_compatibility_version = str(payload["runtime_compatibility_version"])

    if checkpoint_format_version != EXPECTED_TFT_CHECKPOINT_FORMAT_VERSION:
        raise TFTCompatibilityError(
            "Installed TFT checkpoint format version is incompatible. "
            f"Expected {EXPECTED_TFT_CHECKPOINT_FORMAT_VERSION!r}, "
            f"got {checkpoint_format_version!r}."
        )
    if runtime_compatibility_version != EXPECTED_TFT_RUNTIME_COMPATIBILITY_VERSION:
        raise TFTCompatibilityError(
            "Installed TFT runtime compatibility version is incompatible. "
            f"Expected {EXPECTED_TFT_RUNTIME_COMPATIBILITY_VERSION!r}, got "
            f"{runtime_compatibility_version!r}."
        )

    checkpoints: dict[int, Path] = {}
    for horizon_text, relative_path in payload["horizons"].items():
        if not isinstance(relative_path, str):
            raise TFTCheckpointMapError(
                f"Checkpoint path for horizon {horizon_text!r} must be a string."
            )
        horizon = int(horizon_text)
        resolved_checkpoint_path = _resolve_relative_checkpoint_path(
            resolved_artifact_root,
            relative_path,
        )
        if not resolved_checkpoint_path.exists():
            raise TFTExtractionError(
                "TFT checkpoint file listed in the checkpoint map is missing: "
                f"{resolved_checkpoint_path}."
            )
        checkpoints[horizon] = resolved_checkpoint_path

    bundle_url = os.environ.get(TFT_BUNDLE_URL_ENV_VAR)

    return InstalledTFTCheckpointMap(
        artifact_root=resolved_artifact_root,
        bundle_path=resolved_bundle_path,
        manifest_path=manifest_path,
        checkpoints=checkpoints,
        artifact_type=str(payload["artifact_type"]),
        build_version=str(payload["build_version"]),
        checkpoint_format_version=checkpoint_format_version,
        runtime_compatibility_version=runtime_compatibility_version,
        bundle_url=bundle_url,
    )


def load_tft_model_for_horizon(  # pylint: disable=too-many-arguments
    horizon: int,
    *,
    artifact_root: str | Path | None = None,
    bundle_path: str | Path | None = None,
    bundle_version: str | None = None,
    loader: Callable[[Path], Any] | None = None,
    dependency_module: str = "torch",
) -> Any:
    """Load the offline-trained TFT model for one requested horizon."""
    validated_horizon = validate_horizon(horizon)
    installed_map = load_installed_tft_checkpoint_map(
        artifact_root=artifact_root,
        bundle_path=bundle_path,
        bundle_version=bundle_version,
    )

    if loader is None:
        dependency = _import_tft_dependency(dependency_module)
        loader = dependency.load

    return loader(installed_map.checkpoints[validated_horizon])


def predict_tft(  # pylint: disable=too-many-arguments
    df: pd.DataFrame,
    horizon: int,
    *,
    model: Any | None = None,
    artifact_root: str | Path | None = None,
    bundle_path: str | Path | None = None,
    bundle_version: str | None = None,
    loader: Callable[[Path], Any] | None = None,
    dependency_module: str = "torch",
) -> pd.Series:
    """Return TFT predictions aligned to the input DataFrame index."""
    validated_horizon = validate_horizon(horizon)
    resolved_model = model
    if resolved_model is None:
        resolved_model = load_tft_model_for_horizon(
            validated_horizon,
            artifact_root=artifact_root,
            bundle_path=bundle_path,
            bundle_version=bundle_version,
            loader=loader,
            dependency_module=dependency_module,
        )

    if hasattr(resolved_model, "predict"):
        predictions = resolved_model.predict(df)
    elif callable(resolved_model):
        predictions = resolved_model(df)
    else:
        raise TypeError(
            "Loaded TFT model must be callable or provide a predict(df) method."
        )

    prediction_series = pd.Series(predictions, index=df.index, name="prediction")
    if len(prediction_series) != len(df):
        raise ValueError(
            "TFT prediction output must align one-to-one with the input DataFrame rows."
        )
    return prediction_series


def evaluate_tft_model(  # pylint: disable=too-many-arguments
    df: pd.DataFrame,
    horizon: int,
    *,
    model: Any | None = None,
    artifact_root: str | Path | None = None,
    bundle_path: str | Path | None = None,
    bundle_version: str | None = None,
    loader: Callable[[Path], Any] | None = None,
    dependency_module: str = "torch",
) -> pd.DataFrame:
    """Return a shared-schema evaluation table for TFT inference."""
    validated_horizon = validate_horizon(horizon)
    predictions = predict_tft(
        df,
        validated_horizon,
        model=model,
        artifact_root=artifact_root,
        bundle_path=bundle_path,
        bundle_version=bundle_version,
        loader=loader,
        dependency_module=dependency_module,
    )
    target_column = get_target_column(validated_horizon)
    return build_results_table(
        df.loc[:, ["zone"]],
        df[target_column],
        predictions,
        model_name="tft",
        horizon=validated_horizon,
    )
