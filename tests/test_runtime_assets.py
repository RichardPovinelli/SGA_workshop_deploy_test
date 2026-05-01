# pylint: disable=missing-module-docstring, missing-function-docstring

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from src.config import (
    EXPECTED_DATASET_RUNTIME_COMPATIBILITY_VERSION,
    EXPECTED_TFT_RUNTIME_COMPATIBILITY_VERSION,
    TFT_BUNDLE_FILENAME,
)
from src.runtime_assets import (
    DATASET_INSTALL_STATE_PATH,
    RELEASE_ASSETS_MANIFEST_FILENAME,
    TFT_INSTALL_STATE_PATH,
    build_dataset_asset_filename,
    build_github_release_asset_url,
    ensure_supported_bootstrap_platform,
    get_dataset_asset_descriptor,
    get_tft_asset_descriptor,
    install_dataset_from_file,
    install_tft_bundle_from_file,
    load_install_state,
    validate_dataset_manifest_payload,
    write_release_assets_manifest,
)


def test_asset_descriptor_construction_uses_tag_based_contract() -> None:
    dataset = get_dataset_asset_descriptor()
    tft = get_tft_asset_descriptor()

    assert dataset.artifact_type == "dataset"
    assert tft.artifact_type == "tft_checkpoints"
    assert dataset.release_tag != tft.release_tag
    assert dataset.filename.endswith(".parquet.gz")
    assert tft.filename.endswith(".tgz")


def test_build_release_asset_url_uses_owner_repo_tag_and_filename() -> None:
    url = build_github_release_asset_url("owner", "repo", "tag-1", "file.txt")
    assert url == "https://github.com/owner/repo/releases/download/tag-1/file.txt"


def test_validate_dataset_manifest_payload_rejects_missing_fields() -> None:
    with pytest.raises(ValueError, match="missing required field"):
        validate_dataset_manifest_payload({"artifact_type": "dataset"})


def test_supported_bootstrap_platform_accepts_linux_and_rejects_windows() -> None:
    assert ensure_supported_bootstrap_platform("Linux") == "Linux"
    with pytest.raises(RuntimeError, match="POSIX-style environments"):
        ensure_supported_bootstrap_platform("Windows")


def test_write_release_assets_manifest_contains_both_streams(tmp_path: Path) -> None:
    manifest_path = write_release_assets_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.name == RELEASE_ASSETS_MANIFEST_FILENAME
    assert set(payload) == {"dataset", "tft_checkpoints"}
    assert payload["dataset"]["artifact_type"] == "dataset"
    assert payload["tft_checkpoints"]["artifact_type"] == "tft_checkpoints"


def test_install_dataset_from_file_writes_install_state(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = tmp_path / build_dataset_asset_filename("1")
    manifest_path = tmp_path / "dataset_manifest.json"
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "zone": ["zone_a", "zone_a", "zone_b"],
            "split": ["train", "train", "test"],
            "demand": [1.0, 2.0, 3.0],
            "temp": [60.0, 61.0, 62.0],
            "hdd": [5.0, 4.0, 3.0],
            "cdd": [0.0, 0.0, 0.0],
            "dow": [0.0, 1.0, 2.0],
            "is_weekend": [0.0, 0.0, 0.0],
            "month": [1.0, 1.0, 1.0],
            "lag1": [0.5, 1.0, 2.0],
            "lag7": [0.5, 0.6, 0.7],
            "target_h1": [2.0, 3.0, 4.0],
            "target_h2": [3.0, 4.0, 5.0],
            "target_h3": [4.0, 5.0, 6.0],
            "target_h4": [5.0, 6.0, 7.0],
            "target_h5": [6.0, 7.0, 8.0],
            "target_h6": [7.0, 8.0, 9.0],
            "target_h7": [8.0, 9.0, 10.0],
        }
    )
    df.to_parquet(dataset_path, engine="pyarrow", compression="gzip")
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "dataset",
                "artifact_filename": dataset_path.name,
                "schema_version": "1",
                "runtime_compatibility_version": EXPECTED_DATASET_RUNTIME_COMPATIBILITY_VERSION,
                "build_version": "1",
                "release_tag": "dataset-v1",
                "rows": 3,
                "columns": list(df.columns),
                "zone_list": ["zone_a", "zone_b"],
                "min_date": "2024-01-01",
                "max_date": "2024-01-03",
                "split_counts": {"test": 1, "train": 2},
                "build_seed": 1,
                "build_timestamp": "2026-04-22T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    local_state_root = tmp_path / "local_state"
    monkeypatch.setenv("SGA_WORKSHOP_LOCAL_STATE_ROOT", str(local_state_root))

    install_path = install_dataset_from_file(
        dataset_path, source_manifest_path=manifest_path
    )
    install_state = load_install_state(DATASET_INSTALL_STATE_PATH)

    assert install_path.exists()
    assert install_state is not None
    assert install_state["artifact_type"] == "dataset"


def test_install_tft_bundle_from_file_writes_install_state(
    tmp_path: Path, monkeypatch
) -> None:
    local_state_root = tmp_path / "local_state"
    monkeypatch.setenv("SGA_WORKSHOP_LOCAL_STATE_ROOT", str(local_state_root))

    bundle_source_root = tmp_path / "bundle_root"
    checkpoints_dir = bundle_source_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    for horizon in range(1, 8):
        (checkpoints_dir / f"h{horizon}.ckpt").write_text("ok\n", encoding="utf-8")
    (bundle_source_root / "checkpoint_map.json").write_text(
        json.dumps(
            {
                "artifact_type": "tft_checkpoints",
                "build_version": "1",
                "checkpoint_format_version": "1",
                "runtime_compatibility_version": EXPECTED_TFT_RUNTIME_COMPATIBILITY_VERSION,
                "supported_horizons": [1, 2, 3, 4, 5, 6, 7],
                "horizons": {
                    str(horizon): f"checkpoints/h{horizon}.ckpt"
                    for horizon in range(1, 8)
                },
            }
        ),
        encoding="utf-8",
    )
    bundle_path = tmp_path / TFT_BUNDLE_FILENAME
    with tarfile.open(bundle_path, "w:gz") as archive:
        for child in sorted(bundle_source_root.iterdir()):
            archive.add(child, arcname=child.name)

    extract_root = install_tft_bundle_from_file(
        bundle_path, requested_filename=bundle_path.name
    )
    install_state = load_install_state(TFT_INSTALL_STATE_PATH)

    assert extract_root.exists()
    assert (extract_root / "checkpoint_map.json").exists()
    assert install_state is not None
    assert install_state["artifact_type"] == "tft_checkpoints"
