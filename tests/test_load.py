# pylint: disable=missing-module-docstring, missing-function-docstring

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from tests.test_schema import make_valid_df
from src.data.load import load_dataset


def test_load_dataset_reads_parquet_and_parses_dates(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    manifest_path = tmp_path / "dataset_manifest.json"
    expected = make_valid_df()
    expected.to_parquet(dataset_path, engine="pyarrow")
    manifest_path.write_text(
        (
            "{"
            '"artifact_type":"dataset",'
            '"artifact_filename":"dataset.parquet",'
            '"schema_version":"1",'
            '"runtime_compatibility_version":"1",'
            '"build_version":"1",'
            '"release_tag":"dataset-v1",'
            f'"rows":{len(expected)},'
            f'"columns":{list(expected.columns)!r},'
            '"zone_list":["zone_a","zone_b"],'
            '"min_date":"2024-01-01",'
            '"max_date":"2024-01-03",'
            '"split_counts":{"test":1,"train":2},'
            '"build_seed":1,'
            '"build_timestamp":"2026-04-22T00:00:00+00:00"'
            "}"
        ).replace("'", '"'),
        encoding="utf-8",
    )

    actual = load_dataset(dataset_path, manifest_path=manifest_path)

    assert pd.api.types.is_datetime64_any_dtype(actual["date"])
    pdt.assert_frame_equal(actual, expected)


def test_load_dataset_reads_parquet_gz(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet.gz"
    manifest_path = tmp_path / "dataset_manifest.json"
    expected = make_valid_df()
    expected.to_parquet(dataset_path, engine="pyarrow", compression="gzip")
    manifest_path.write_text(
        (
            "{"
            '"artifact_type":"dataset",'
            '"artifact_filename":"dataset.parquet.gz",'
            '"schema_version":"1",'
            '"runtime_compatibility_version":"1",'
            '"build_version":"1",'
            '"release_tag":"dataset-v1",'
            f'"rows":{len(expected)},'
            f'"columns":{list(expected.columns)!r},'
            '"zone_list":["zone_a","zone_b"],'
            '"min_date":"2024-01-01",'
            '"max_date":"2024-01-03",'
            '"split_counts":{"test":1,"train":2},'
            '"build_seed":1,'
            '"build_timestamp":"2026-04-22T00:00:00+00:00"'
            "}"
        ).replace("'", '"'),
        encoding="utf-8",
    )

    actual = load_dataset(dataset_path, manifest_path=manifest_path)

    pdt.assert_frame_equal(actual, expected)


def test_load_dataset_rejects_invalid_extension(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("not-used\n", encoding="utf-8")

    with pytest.raises(ValueError, match=".parquet"):
        load_dataset(dataset_path)
