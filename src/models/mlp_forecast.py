"""Inference-only MLP checkpoint loading and prediction helpers."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn

    class _MLP(nn.Module):
        """Simple MLP for direct horizon forecasting."""

        def __init__(self, n_features: int, neurons_in_hidden_layers: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, neurons_in_hidden_layers),
                nn.ReLU(),
                nn.Linear(neurons_in_hidden_layers, neurons_in_hidden_layers),
                nn.ReLU(),
                nn.Linear(neurons_in_hidden_layers, neurons_in_hidden_layers),
                nn.ReLU(),
                nn.Linear(neurons_in_hidden_layers, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x).squeeze(-1)

except ImportError:
    pass

from src.config import (
    ALLOWED_HORIZONS,
    DEFAULT_MLP_FORECAST_BUNDLE_DIR,
    DEFAULT_MLP_FORECAST_BUNDLE_VERSION,
    DEFAULT_MLP_FORECAST_INSTALL_ROOT,
    EXPECTED_MLP_FORECAST_CHECKPOINT_FORMAT_VERSION,
    EXPECTED_MLP_FORECAST_RUNTIME_COMPATIBILITY_VERSION,
    SUPPORTED_MLP_FORECAST_HORIZONS,
    MLP_FORECAST_ARTIFACT_TYPE,
    MLP_FORECAST_BUNDLE_FILENAME_TEMPLATE,
    MLP_FORECAST_BUNDLE_URL_ENV_VAR,
    MLP_FORECAST_CHECKPOINT_MAP_FILENAME,
)
from src.data.split import (
    get_feature_columns,
    get_target_column,
    get_train_df,
    validate_horizon,
)
from src.evaluation.metrics import build_results_table


class MLPForecastHorizonModel:
    """Trained horizon model satisfying the workshop predict(df) contract."""

    def __init__(self, net: Any, feature_columns: list[str], x_mean: np.ndarray, x_std: np.ndarray) -> None:
        self.net = net
        self.feature_columns = feature_columns
        self.x_mean = x_mean
        self.x_std = x_std

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        import torch

        x = df[self.feature_columns].to_numpy(dtype=np.float32)
        x = (x - self.x_mean) / (self.x_std + 1e-8)
        tensor = torch.tensor(x, dtype=torch.float32)
        self.net.eval()
        with torch.no_grad():
            return self.net(tensor).numpy()


class MLPForecastDependencyError(ImportError):
    """Raised when the runtime MLP Forecast dependency is not available."""


class MLPForecastBundleNotFoundError(FileNotFoundError):
    """Raised when the bundled MLP Forecast checkpoint asset is missing."""


class MLPForecastExtractionError(FileNotFoundError):
    """Raised when extracted MLP Forecast files are missing or incomplete."""


class MLPForecastCheckpointMapError(ValueError):
    """Raised when the checkpoint map is malformed."""


class MLPForecastCompatibilityError(ValueError):
    """Raised when installed MLP Forecast artifact metadata is incompatible."""


def train_mlp_forecast_models(
    df: pd.DataFrame | None = None,
    *,
    epochs: int = 100,
    neurons_in_hidden_layers: int = 64,
    seed: int = 0,
) -> dict[int, MLPForecastHorizonModel]:
    """Train MLP models for all 7 horizons and return dict of trained models.

    Args:
        df: Input DataFrame. If None, loads from dataset.
        epochs: Training epochs per horizon (default: 100)
        neurons_in_hidden_layers: Hidden layer size (default: 64)
        seed: Random seed (default: 0)

    Returns:
        Dict mapping horizon (1-7) to trained MLPForecastHorizonModel
    """
    _import_mlp_forecast_dependency("torch")
    from torch.utils.data import DataLoader, TensorDataset  # pylint: disable=import-outside-toplevel
    from src.data.load import load_dataset  # pylint: disable=import-outside-toplevel

    resolved_df = df if df is not None else load_dataset()
    trained_models: dict[int, MLPForecastHorizonModel] = {}

    for horizon in ALLOWED_HORIZONS:
        torch.manual_seed(seed + horizon)
        np.random.seed(seed + horizon)

        feature_columns = get_feature_columns(horizon)
        train_df = get_train_df(resolved_df)
        target_col = get_target_column(horizon)

        x = train_df[feature_columns].to_numpy(dtype=np.float32)
        y = train_df[target_col].to_numpy(dtype=np.float32)

        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x_norm = (x - x_mean) / (x_std + 1e-8)

        dataset = TensorDataset(
            torch.tensor(x_norm, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        net = _MLP(n_features=len(feature_columns), neurons_in_hidden_layers=neurons_in_hidden_layers)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        net.train()
        for epoch in range(epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = loss_fn(net(x_batch), y_batch)
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"  Horizon {horizon}, epoch {epoch + 1}/{epochs}  loss={loss.item():.4f}")

        trained_models[horizon] = MLPForecastHorizonModel(
            net=net, feature_columns=feature_columns, x_mean=x_mean, x_std=x_std
        )
        print(f"✓ Horizon {horizon} trained")

    return trained_models


@dataclass(frozen=True)
class InstalledMLPForecastCheckpointMap:  # pylint: disable=too-many-instance-attributes
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


def build_mlp_forecast_checkpoint_bundle_filename(version: str) -> str:
    """Return the canonical MLP Forecast checkpoint bundle filename for a release version."""
    if not version or version.strip() == "":
        raise ValueError("version must be a non-empty string.")
    return MLP_FORECAST_BUNDLE_FILENAME_TEMPLATE.format(version=version)


def _resolve_mlp_forecast_bundle_path(
    *,
    bundle_path: str | Path | None = None,
    bundle_version: str | None = None,
) -> Path:
    if bundle_path is not None:
        return Path(bundle_path)
    resolved_version = bundle_version or DEFAULT_MLP_FORECAST_BUNDLE_VERSION
    return DEFAULT_MLP_FORECAST_BUNDLE_DIR / build_mlp_forecast_checkpoint_bundle_filename(resolved_version)


def _resolve_mlp_forecast_artifact_root(artifact_root: str | Path | None = None) -> Path:
    return Path(artifact_root) if artifact_root is not None else DEFAULT_MLP_FORECAST_INSTALL_ROOT


def _import_mlp_forecast_dependency(module_name: str = "torch") -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise MLPForecastDependencyError(
            "MLP Forecast inference dependencies are missing. Install the required MLP Forecast runtime "
            f"package(s) for the workshop environment before loading checkpoints: {module_name!r}."
        ) from exc


def _validate_checkpoint_map_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise MLPForecastCheckpointMapError("MLP Forecast checkpoint map must be a JSON object.")

    required_top_level_fields = (
        "artifact_type",
        "build_version",
        "checkpoint_format_version",
        "runtime_compatibility_version",
        "supported_horizons",
        "horizons",
    )
    missing_fields = [field for field in required_top_level_fields if field not in payload]
    if missing_fields:
        raise MLPForecastCheckpointMapError(
            f"MLP Forecast checkpoint map is missing required field(s): {missing_fields}."
        )

    horizons = payload["horizons"]
    if not isinstance(horizons, dict):
        raise MLPForecastCheckpointMapError("MLP Forecast checkpoint map field 'horizons' must be an object.")
    if payload["artifact_type"] != MLP_FORECAST_ARTIFACT_TYPE:
        raise MLPForecastCheckpointMapError(
            "MLP Forecast checkpoint map artifact_type must be 'mlp_forecast_checkpoints'."
        )

    expected_keys = {str(horizon) for horizon in ALLOWED_HORIZONS}
    observed_keys = set(horizons.keys())
    if observed_keys != expected_keys:
        raise MLPForecastCheckpointMapError(
            f"MLP Forecast checkpoint map must define horizons 1 through 7 exactly; found keys: {sorted(observed_keys)}."
        )
    supported_horizons = tuple(payload["supported_horizons"])
    if tuple(int(horizon) for horizon in supported_horizons) != SUPPORTED_MLP_FORECAST_HORIZONS:
        raise MLPForecastCheckpointMapError(
            "MLP Forecast checkpoint map supported_horizons must match 1 through 7 exactly."
        )


def _resolve_relative_checkpoint_path(artifact_root: Path, relative_path: str) -> Path:
    checkpoint_path = Path(relative_path)
    if checkpoint_path.is_absolute():
        raise MLPForecastCheckpointMapError("Checkpoint paths must be relative to the extracted MLP Forecast root.")
    if not relative_path or relative_path.strip() == "":
        raise MLPForecastCheckpointMapError("Checkpoint paths must be non-empty relative paths.")
    if ".." in checkpoint_path.parts:
        raise MLPForecastCheckpointMapError(
            "Checkpoint paths must stay within the extracted MLP Forecast root; '..' is not allowed."
        )
    return artifact_root / checkpoint_path


def load_installed_mlp_forecast_checkpoint_map(  # pylint: disable=too-many-locals
    *,
    artifact_root: str | Path | None = None,
    bundle_path: str | Path | None = None,
    bundle_version: str | None = None,
) -> InstalledMLPForecastCheckpointMap:
    """Load and validate the installed MLP Forecast checkpoint layout."""
    resolved_artifact_root = _resolve_mlp_forecast_artifact_root(artifact_root)
    resolved_bundle_path = _resolve_mlp_forecast_bundle_path(
        bundle_path=bundle_path,
        bundle_version=bundle_version,
    )

    if not resolved_artifact_root.exists():
        if not resolved_bundle_path.exists():
            raise MLPForecastBundleNotFoundError(
                f"MLP Forecast checkpoint bundle is missing. Expected bundle asset at {resolved_bundle_path}."
            )
        raise MLPForecastExtractionError(
            f"MLP Forecast checkpoint bundle was found but the extracted checkpoint root is missing: {resolved_artifact_root}."
        )

    manifest_path = resolved_artifact_root / MLP_FORECAST_CHECKPOINT_MAP_FILENAME
    if not manifest_path.exists():
        raise MLPForecastExtractionError(
            "Extracted MLP Forecast checkpoint root is missing the required manifest "
            f"{MLP_FORECAST_CHECKPOINT_MAP_FILENAME}: {manifest_path}."
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MLPForecastCheckpointMapError(f"MLP Forecast checkpoint map is not valid JSON: {manifest_path}.") from exc

    _validate_checkpoint_map_payload(payload)

    checkpoint_format_version = str(payload["checkpoint_format_version"])
    runtime_compatibility_version = str(payload["runtime_compatibility_version"])

    if checkpoint_format_version != EXPECTED_MLP_FORECAST_CHECKPOINT_FORMAT_VERSION:
        raise MLPForecastCompatibilityError(
            "Installed MLP Forecast checkpoint format version is incompatible. "
            f"Expected {EXPECTED_MLP_FORECAST_CHECKPOINT_FORMAT_VERSION!r}, "
            f"got {checkpoint_format_version!r}."
        )
    if runtime_compatibility_version != EXPECTED_MLP_FORECAST_RUNTIME_COMPATIBILITY_VERSION:
        raise MLPForecastCompatibilityError(
            "Installed MLP Forecast runtime compatibility version is incompatible. "
            f"Expected {EXPECTED_MLP_FORECAST_RUNTIME_COMPATIBILITY_VERSION!r}, got "
            f"{runtime_compatibility_version!r}."
        )

    checkpoints: dict[int, Path] = {}
    for horizon_text, relative_path in payload["horizons"].items():
        if not isinstance(relative_path, str):
            raise MLPForecastCheckpointMapError(f"Checkpoint path for horizon {horizon_text!r} must be a string.")
        horizon = int(horizon_text)
        resolved_checkpoint_path = _resolve_relative_checkpoint_path(
            resolved_artifact_root,
            relative_path,
        )
        if not resolved_checkpoint_path.exists():
            raise MLPForecastExtractionError(
                f"MLP Forecast checkpoint file listed in the checkpoint map is missing: {resolved_checkpoint_path}."
            )
        checkpoints[horizon] = resolved_checkpoint_path

    bundle_url = os.environ.get(MLP_FORECAST_BUNDLE_URL_ENV_VAR)

    return InstalledMLPForecastCheckpointMap(
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


def load_mlp_forecast_model_for_horizon(  # pylint: disable=too-many-arguments
    horizon: int,
    *,
    artifact_root: str | Path | None = None,
    bundle_path: str | Path | None = None,
    bundle_version: str | None = None,
    loader: Callable[[Path], Any] | None = None,
    dependency_module: str = "torch",
) -> Any:
    """Load the offline-trained MLP Forecast model for one requested horizon."""
    validated_horizon = validate_horizon(horizon)
    installed_map = load_installed_mlp_forecast_checkpoint_map(
        artifact_root=artifact_root,
        bundle_path=bundle_path,
        bundle_version=bundle_version,
    )

    if loader is None:
        dependency = _import_mlp_forecast_dependency(dependency_module)
        # weights_only=False required for checkpoints saved before PyTorch 2.6
        loader = lambda path: dependency.load(path, weights_only=False)  # noqa: E731

    assert loader is not None
    return loader(installed_map.checkpoints[validated_horizon])


def predict_mlp_forecast(
    df: pd.DataFrame,
    horizon: int,
    *,
    model: Any,
) -> pd.Series:
    """Return MLP Forecast predictions aligned to the input DataFrame index.

    Args:
        df: Input DataFrame for predictions
        horizon: Forecast horizon (1-7)
        model: Trained MLPForecastHorizonModel instance (required)

    Returns:
        Predictions as Series aligned to df.index
    """
    validated_horizon = validate_horizon(horizon)

    if not hasattr(model, "predict"):
        raise TypeError("Model must have a predict(df) method.")

    predictions = model.predict(df)
    prediction_series = pd.Series(predictions, index=df.index, name="prediction")
    if len(prediction_series) != len(df):
        raise ValueError("MLP Forecast prediction output must align one-to-one with the input DataFrame rows.")
    return prediction_series


def evaluate_mlp_forecast_model(
    df: pd.DataFrame,
    horizon: int,
    *,
    model: Any,
) -> pd.DataFrame:
    """Return a shared-schema evaluation table for MLP Forecast inference.

    Args:
        df: Input DataFrame with target column
        horizon: Forecast horizon (1-7)
        model: Trained MLPForecastHorizonModel instance (required)

    Returns:
        Evaluation results table with predictions and metrics
    """
    validated_horizon = validate_horizon(horizon)
    predictions = predict_mlp_forecast(df, validated_horizon, model=model)
    target_column = get_target_column(validated_horizon)
    return build_results_table(
        df[target_column],
        predictions,
        model_name="mlp_forecast",
        horizon=validated_horizon,
    )
