"""Temporal Fusion Transformer model for multi-step demand forecasting."""

from __future__ import annotations

import logging
import json
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    ALLOWED_HORIZONS,
    DEFAULT_TFT_ARTIFACT_ROOT,
    EXPECTED_TFT_CHECKPOINT_FORMAT_VERSION,
    EXPECTED_TFT_RUNTIME_COMPATIBILITY_VERSION,
    TFT_ARTIFACT_TYPE,
    TFT_CHECKPOINT_MAP_FILENAME,
    TFT_MAX_ENCODER_LENGTH,
    TFT_MAX_PREDICTION_LENGTH,
)
from src.data.split import get_test_df, get_train_df, validate_horizon
from src.evaluation.metrics import build_results_table


class TFTDependencyError(RuntimeError):
    """pytorch-forecasting or lightning is not installed."""


class TFTBundleNotFoundError(FileNotFoundError):
    """TFT checkpoint directory or file is missing."""


class TFTCheckpointMapError(ValueError):
    """checkpoint_map.json is missing, malformed, or has wrong structure."""


class TFTCompatibilityError(RuntimeError):
    """Checkpoint version is incompatible with the current runtime."""


class TFTExtractionError(RuntimeError):
    """Checkpoint could not be loaded."""


@dataclass(frozen=True)
class InstalledTFTCheckpointMap:
    """Resolved checkpoint layout for a TFT artifact root."""

    artifact_root: Path
    checkpoint_path: Path
    manifest_path: Path
    artifact_type: str
    checkpoint_format_version: str
    runtime_compatibility_version: str
    supported_horizons: tuple[int, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TFT_KNOWN_REALS = ["time_idx", "dow", "is_weekend", "month"]
_TFT_UNKNOWN_REALS = ["demand", "temp", "hdd", "cdd"]


@contextmanager
def _suppress_tft_runtime_noise() -> Any:
    """Suppress Lightning-family warnings/log output during TFT operations."""
    logger_names = (
        "lightning",
        "lightning.pytorch",
        "lightning_fabric",
        "lightning_utilities",
        "pytorch_lightning",
        "pytorch_forecasting",
    )
    previous_levels: dict[str, int] = {}
    for name in logger_names:
        logger = logging.getLogger(name)
        previous_levels[name] = logger.level
        # CRITICAL+1 effectively silences all records for this logger tree.
        logger.setLevel(logging.CRITICAL + 1)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module=r"(lightning|lightning_utilities|lightning_fabric|pytorch_lightning|pytorch_forecasting)(\..*)?",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"(lightning|lightning_utilities|lightning_fabric|pytorch_lightning|pytorch_forecasting)(\..*)?",
            )
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                module=r"(lightning|lightning_utilities|lightning_fabric|pytorch_lightning|pytorch_forecasting)(\..*)?",
            )
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                module=r"(lightning|lightning_utilities|lightning_fabric|pytorch_lightning|pytorch_forecasting)(\..*)?",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*isinstance\(treespec, LeafSpec\).*deprecated.*",
                category=Warning,
                module=r"torch\.utils\._pytree",
            )
            yield
    finally:
        for name in logger_names:
            logging.getLogger(name).setLevel(previous_levels[name])


def _prepare_tft_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date and add time_idx / group_id columns required by TimeSeriesDataSet."""
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["time_idx"] = range(len(df))
    df["group_id"] = "all"
    for col in ("dow", "is_weekend", "month"):
        df[col] = df[col].astype(float)
    return df


def _make_tft_dataset(
    df: pd.DataFrame,
    *,
    max_encoder_length: int,
    max_prediction_length: int,
    predict_mode: bool = False,
) -> Any:
    try:
        from pytorch_forecasting import TimeSeriesDataSet
    except ImportError as exc:
        raise TFTDependencyError("pytorch-forecasting is required: pip install pytorch-forecasting") from exc

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="demand",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=_TFT_KNOWN_REALS,
        time_varying_unknown_reals=_TFT_UNKNOWN_REALS,
        allow_missing_timesteps=False,
        predict_mode=predict_mode,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_tft_model(
    df: pd.DataFrame | None = None,
    *,
    max_encoder_length: int = TFT_MAX_ENCODER_LENGTH,
    max_epochs: int = 50,
    seed: int = 0,
) -> Any:
    """Train a TFT model on the training split and return the fitted model."""
    try:
        import lightning.pytorch as pl
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import MAE
    except ImportError as exc:
        raise TFTDependencyError(
            "pytorch-forecasting and lightning are required: pip install pytorch-forecasting lightning"
        ) from exc

    if df is None:
        from src.data.load import load_dataset

        df = load_dataset()

    pl.seed_everything(seed)

    train_df = _prepare_tft_dataframe(get_train_df(df))

    # Use last 10% of training rows as validation
    cutoff = int(len(train_df) * 0.9)
    train_data = train_df.iloc[:cutoff]
    val_data = train_df.iloc[cutoff:]

    training_dataset = _make_tft_dataset(
        train_data,
        max_encoder_length=max_encoder_length,
        max_prediction_length=TFT_MAX_PREDICTION_LENGTH,
    )
    validation_dataset = _make_tft_dataset(
        val_data,
        max_encoder_length=max_encoder_length,
        max_prediction_length=TFT_MAX_PREDICTION_LENGTH,
    )

    train_loader = training_dataset.to_dataloader(train=True, batch_size=64, num_workers=15)
    val_loader = validation_dataset.to_dataloader(train=False, batch_size=64, num_workers=15)

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=MAE(),
        reduce_on_plateau_patience=4,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_model_summary=False,
        enable_progress_bar=True,
        logger=False,
        gradient_clip_val=0.1,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    return tft


# ---------------------------------------------------------------------------
# Checkpoint map
# ---------------------------------------------------------------------------


def load_installed_tft_checkpoint_map(
    *,
    artifact_root: Path | None = None,
) -> InstalledTFTCheckpointMap:
    """Load and validate the TFT checkpoint map from the artifact root."""
    root = Path(artifact_root) if artifact_root is not None else DEFAULT_TFT_ARTIFACT_ROOT

    if not root.exists():
        raise TFTBundleNotFoundError(f"TFT artifact root not found: {root}")

    manifest_path = root / TFT_CHECKPOINT_MAP_FILENAME
    if not manifest_path.exists():
        raise TFTCheckpointMapError(f"checkpoint_map.json not found at {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TFTCheckpointMapError(f"checkpoint_map.json is not valid JSON: {exc}") from exc

    artifact_type = manifest.get("artifact_type")
    if artifact_type != TFT_ARTIFACT_TYPE:
        raise TFTCheckpointMapError(f"Expected artifact_type {TFT_ARTIFACT_TYPE!r}, got {artifact_type!r}")

    fmt_version = manifest.get("checkpoint_format_version")
    if fmt_version != EXPECTED_TFT_CHECKPOINT_FORMAT_VERSION:
        raise TFTCompatibilityError(
            f"Checkpoint format version {fmt_version!r} is incompatible "
            f"(expected {EXPECTED_TFT_CHECKPOINT_FORMAT_VERSION!r})"
        )

    rt_version = manifest.get("runtime_compatibility_version")
    if rt_version != EXPECTED_TFT_RUNTIME_COMPATIBILITY_VERSION:
        raise TFTCompatibilityError(
            f"Runtime compatibility version {rt_version!r} is incompatible "
            f"(expected {EXPECTED_TFT_RUNTIME_COMPATIBILITY_VERSION!r})"
        )

    checkpoint_rel = manifest.get("checkpoint")
    if not checkpoint_rel:
        raise TFTCheckpointMapError("checkpoint_map.json is missing the 'checkpoint' key")

    checkpoint_path = root / checkpoint_rel
    if not checkpoint_path.exists():
        raise TFTBundleNotFoundError(f"TFT checkpoint not found: {checkpoint_path}")

    supported_horizons = tuple(manifest.get("supported_horizons", ALLOWED_HORIZONS))

    return InstalledTFTCheckpointMap(
        artifact_root=root,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        artifact_type=artifact_type,
        checkpoint_format_version=fmt_version,
        runtime_compatibility_version=rt_version,
        supported_horizons=supported_horizons,
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_tft_model(
    *,
    artifact_root: Path | None = None,
) -> Any:
    """Load the TFT model from the artifact root checkpoint."""
    try:
        from pytorch_forecasting import TemporalFusionTransformer
    except ImportError as exc:
        raise TFTDependencyError("pytorch-forecasting is required: pip install pytorch-forecasting") from exc

    checkpoint_map = load_installed_tft_checkpoint_map(artifact_root=artifact_root)

    try:
        with _suppress_tft_runtime_noise():
            return TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_map.checkpoint_path))
    except Exception as exc:
        raise TFTExtractionError(f"Failed to load TFT checkpoint from {checkpoint_map.checkpoint_path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _predict_all_horizons(
    df: pd.DataFrame,
    *,
    model: Any,
    max_encoder_length: int = TFT_MAX_ENCODER_LENGTH,
) -> tuple[np.ndarray, pd.Index]:
    """Return (predictions, test_index) where predictions has shape (n_test, 7).

    Predictions are ordered to match the test rows in df. Rows lacking 7 future
    context rows in df will have NaN predictions.
    """
    df_prep = _prepare_tft_dataframe(df)
    test_mask = df_prep["split"] == "test"
    test_rows = df_prep[test_mask]
    n_test = len(test_rows)
    original_test_index = df[df["split"] == "test"].index
    all_preds = np.full((n_test, TFT_MAX_PREDICTION_LENGTH), np.nan)

    # Build one window per test row, each tagged with a unique group_id.
    # pytorch-forecasting with predict_mode=True creates exactly one prediction
    # per group (the last valid decoder position), so n_groups == n_predictions.
    # Concatenating all windows into one DataFrame lets us run a single
    # dataset + DataLoader + model.predict() call instead of one per test row.
    windows: list[pd.DataFrame] = []
    valid_indices: list[int] = []  # positions in all_preds that will be filled

    for i, (_, row) in enumerate(test_rows.iterrows()):
        end_idx = int(row["time_idx"])
        start_idx = max(0, end_idx - max_encoder_length + 1)
        future_end_idx = end_idx + TFT_MAX_PREDICTION_LENGTH

        window = df_prep[
            (df_prep["time_idx"] >= start_idx) & (df_prep["time_idx"] <= future_end_idx)
        ].copy()

        if len(window[window["time_idx"] > end_idx]) < TFT_MAX_PREDICTION_LENGTH:
            continue

        window["group_id"] = f"g{i}"
        windows.append(window)
        valid_indices.append(i)

    if not windows:
        return all_preds, original_test_index

    combined_df = pd.concat(windows, ignore_index=True)

    with _suppress_tft_runtime_noise():
        dataset = _make_tft_dataset(
            combined_df,
            max_encoder_length=max_encoder_length,
            max_prediction_length=TFT_MAX_PREDICTION_LENGTH,
            predict_mode=True,
        )
        loader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
        raw = model.predict(loader, mode="prediction")

    raw_np = raw.detach().cpu().numpy()  # shape: (n_valid, 7)

    # Map predictions back to all_preds positions using dataset.index group ordering.
    ds_idx = dataset.index
    if "group_id" in ds_idx.columns:
        group_order = ds_idx["group_id"].tolist()
        gid_to_pred = {gid: raw_np[j] for j, gid in enumerate(group_order)}
        for test_i in valid_indices:
            gid = f"g{test_i}"
            if gid in gid_to_pred:
                all_preds[test_i] = gid_to_pred[gid]
    else:
        # Fallback: output order matches valid_indices order
        for j, test_i in enumerate(valid_indices):
            if j < len(raw_np):
                all_preds[test_i] = raw_np[j]

    return all_preds, original_test_index


def predict_tft(
    df: pd.DataFrame,
    horizon: int,
    *,
    model: Any,
) -> pd.Series:
    """Return h-step-ahead TFT predictions for all test rows as a Series."""
    validate_horizon(horizon)
    all_preds, test_index = _predict_all_horizons(df, model=model)
    return pd.Series(all_preds[:, horizon - 1], index=test_index, name="prediction")


def predict_tft_all_horizons(
    df: pd.DataFrame,
    *,
    model: Any,
) -> pd.DataFrame:
    """Return TFT predictions for all horizons (1..7) on test rows.

    Returns a DataFrame indexed to test rows with columns:
    prediction_h1, prediction_h2, ..., prediction_h7.
    """
    all_preds, test_index = _predict_all_horizons(df, model=model)
    columns = [f"prediction_h{h}" for h in ALLOWED_HORIZONS]
    return pd.DataFrame(all_preds, index=test_index, columns=columns)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_tft_model(
    df: pd.DataFrame,
    horizon: int,
    *,
    model: Any,
) -> pd.DataFrame:
    """Evaluate TFT for a single horizon on the test split.

    Returns a 1-row DataFrame with columns (horizon, model, mape, wmape, rmse).
    """
    from src.data.split import get_target_column

    validate_horizon(horizon)
    test_df = get_test_df(df)
    predictions = predict_tft(df, horizon, model=model)

    target_col = get_target_column(horizon)
    y_true = test_df[target_col]

    # Drop NaN predictions (last few test rows without full future context)
    valid = predictions.notna()
    if not valid.any():
        raise ValueError(
            "All TFT predictions are NaN. This usually means the model diverged during "
            "training (NaN weights). Retrain with: python tools/train_tft_checkpoint.py"
        )
    return build_results_table(
        y_true[valid],
        predictions[valid],
        model_name="tft",
        horizon=horizon,
    )


def evaluate_tft_models(
    df: pd.DataFrame,
    *,
    model: Any,
    predictions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate TFT across all horizons using cached predictions when provided.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset containing train/test split and target_h1..target_h7 columns.
    model : Any
        Loaded TFT model used only when predictions_df is not provided.
    predictions_df : pd.DataFrame | None
        Optional cached predictions from `predict_tft_all_horizons`.

    Returns
    -------
    pd.DataFrame
        Seven-row results table (one per horizon) with columns:
        horizon, model, mape, wmape, rmse.
    """
    from src.data.split import get_target_column, get_test_df

    test_df = get_test_df(df)

    if predictions_df is None:
        predictions_df = predict_tft_all_horizons(df, model=model)

    all_results: list[pd.DataFrame] = []
    for horizon in ALLOWED_HORIZONS:
        target_col = get_target_column(horizon)
        pred_col = f"prediction_h{horizon}"
        if pred_col not in predictions_df.columns:
            raise ValueError(f"predictions_df is missing required column: {pred_col}")

        y_true = test_df[target_col]
        predictions = predictions_df[pred_col]

        valid = predictions.notna()
        if not valid.any():
            raise ValueError(
                "All TFT predictions are NaN for horizon "
                f"{horizon}. This usually means the model diverged during training "
                "(NaN weights). Retrain with: python tools/train_tft_checkpoint.py"
            )

        all_results.append(
            build_results_table(
                y_true[valid],
                predictions[valid],
                model_name="tft",
                horizon=horizon,
            )
        )

    return pd.concat(all_results, ignore_index=True)
