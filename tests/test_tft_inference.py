# pylint: disable=missing-module-docstring, missing-function-docstring
# pylint: disable=missing-class-docstring, too-few-public-methods

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.metrics import RESULTS_TABLE_COLUMNS
from src.models.tft_inference import (
    TFTBundleNotFoundError,
    TFTCheckpointMapError,
    TFTCompatibilityError,
    TFTDependencyError,
    TFTExtractionError,
    build_tft_bundle_filename,
    evaluate_tft_model,
    load_installed_tft_checkpoint_map,
    load_tft_model_for_horizon,
    predict_tft,
)


class MockTFTModel:
    def predict(self, df):
        return [42.0] * len(df)


def _write_checkpoint_bundle(
    artifact_root: Path,
    *,
    build_version: str = "1",
    checkpoint_format_version: str = "1",
    runtime_compatibility_version: str = "1",
    bad_path: str | None = None,
) -> None:
    checkpoints_dir = artifact_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    horizons: dict[str, str] = {}
    for horizon in range(1, 8):
        relative_path = (
            bad_path if bad_path is not None else f"checkpoints/h{horizon}.ckpt"
        )
        horizons[str(horizon)] = relative_path
        if bad_path is None:
            (artifact_root / relative_path).write_text(
                f"horizon={horizon}\n", encoding="utf-8"
            )

    payload = {
        "artifact_type": "tft_checkpoints",
        "build_version": build_version,
        "checkpoint_format_version": checkpoint_format_version,
        "runtime_compatibility_version": runtime_compatibility_version,
        "supported_horizons": [1, 2, 3, 4, 5, 6, 7],
        "horizons": horizons,
    }
    (artifact_root / "checkpoint_map.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_build_tft_bundle_filename_uses_expected_pattern() -> None:
    assert (
        build_tft_bundle_filename("2026.04.22")
        == "sga_workshop_tft_checkpoints_v2026.04.22.tgz"
    )


def test_tft_reports_missing_bundle_when_nothing_is_installed(tmp_path: Path) -> None:
    with pytest.raises(TFTBundleNotFoundError, match="bundle is missing"):
        load_installed_tft_checkpoint_map(
            artifact_root=tmp_path / "installed",
            bundle_path=tmp_path / "bundles" / "missing.tgz",
        )


def test_tft_reports_missing_extraction_separately_from_bundle(tmp_path: Path) -> None:
    bundle_path = tmp_path / "bundles" / "bundle.tgz"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text("placeholder\n", encoding="utf-8")

    with pytest.raises(
        TFTExtractionError, match="extracted checkpoint root is missing"
    ):
        load_installed_tft_checkpoint_map(
            artifact_root=tmp_path / "installed",
            bundle_path=bundle_path,
        )


def test_tft_rejects_malformed_checkpoint_map(tmp_path: Path) -> None:
    artifact_root = tmp_path / "installed"
    artifact_root.mkdir(parents=True)
    (tmp_path / "bundle.tgz").write_text("placeholder\n", encoding="utf-8")
    (artifact_root / "checkpoint_map.json").write_text(
        '{"bad": "payload"}', encoding="utf-8"
    )

    with pytest.raises(TFTCheckpointMapError, match="missing required field"):
        load_installed_tft_checkpoint_map(
            artifact_root=artifact_root,
            bundle_path=tmp_path / "bundle.tgz",
        )


def test_tft_enforces_relative_checkpoint_paths(tmp_path: Path) -> None:
    artifact_root = tmp_path / "installed"
    artifact_root.mkdir(parents=True)
    bundle_path = tmp_path / "bundle.tgz"
    bundle_path.write_text("placeholder\n", encoding="utf-8")
    _write_checkpoint_bundle(artifact_root, bad_path="../escape.ckpt")

    with pytest.raises(
        TFTCheckpointMapError, match="must stay within the extracted TFT root"
    ):
        load_installed_tft_checkpoint_map(
            artifact_root=artifact_root, bundle_path=bundle_path
        )


def test_tft_detects_compatibility_mismatch(tmp_path: Path) -> None:
    artifact_root = tmp_path / "installed"
    artifact_root.mkdir(parents=True)
    bundle_path = tmp_path / "bundle.tgz"
    bundle_path.write_text("placeholder\n", encoding="utf-8")
    _write_checkpoint_bundle(artifact_root, checkpoint_format_version="999")

    with pytest.raises(TFTCompatibilityError, match="format version is incompatible"):
        load_installed_tft_checkpoint_map(
            artifact_root=artifact_root, bundle_path=bundle_path
        )


def test_tft_loader_resolves_all_horizons_and_returns_mocked_model(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "installed"
    artifact_root.mkdir(parents=True)
    bundle_path = tmp_path / "bundle.tgz"
    bundle_path.write_text("placeholder\n", encoding="utf-8")
    _write_checkpoint_bundle(artifact_root)

    seen_paths: list[Path] = []

    def loader(path: Path) -> MockTFTModel:
        seen_paths.append(path)
        return MockTFTModel()

    installed_map = load_installed_tft_checkpoint_map(
        artifact_root=artifact_root,
        bundle_path=bundle_path,
    )
    model = load_tft_model_for_horizon(
        4,
        artifact_root=artifact_root,
        bundle_path=bundle_path,
        loader=loader,
    )

    assert sorted(installed_map.checkpoints) == [1, 2, 3, 4, 5, 6, 7]
    assert seen_paths == [artifact_root / "checkpoints" / "h4.ckpt"]
    assert isinstance(model, MockTFTModel)


def test_tft_dependency_error_is_notebook_friendly(monkeypatch, tmp_path: Path) -> None:
    artifact_root = tmp_path / "installed"
    artifact_root.mkdir(parents=True)
    bundle_path = tmp_path / "bundle.tgz"
    bundle_path.write_text("placeholder\n", encoding="utf-8")
    _write_checkpoint_bundle(artifact_root)

    def fail_dependency(module_name: str = "torch"):
        raise TFTDependencyError("missing torch")

    monkeypatch.setattr(
        "src.models.tft_inference._import_tft_dependency", fail_dependency
    )

    with pytest.raises(TFTDependencyError):
        load_tft_model_for_horizon(
            1, artifact_root=artifact_root, bundle_path=bundle_path
        )


def test_tft_prediction_wrapper_and_results_table_work(
    synthetic_model_df, tmp_path: Path
) -> None:
    test_df = synthetic_model_df.loc[
        (synthetic_model_df["zone"] == "zone_a")
        & (synthetic_model_df["split"] == "test")
    ]
    artifact_root = tmp_path / "installed"
    artifact_root.mkdir(parents=True)
    bundle_path = tmp_path / "bundle.tgz"
    bundle_path.write_text("placeholder\n", encoding="utf-8")
    _write_checkpoint_bundle(artifact_root)

    predictions = predict_tft(test_df, 2, model=MockTFTModel())
    results = evaluate_tft_model(test_df, 2, model=MockTFTModel())

    assert predictions.index.equals(test_df.index)
    assert not predictions.isna().any()
    assert list(results.columns) == list(RESULTS_TABLE_COLUMNS)
    assert results["model"].tolist() == ["tft"]
