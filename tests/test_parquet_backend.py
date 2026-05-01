# pylint: disable=missing-module-docstring, missing-function-docstring

from pathlib import Path

import pandas as pd
import pandas.testing as pdt


def _roundtrip_parquet(dataset_path: Path, compression: str | None) -> None:
    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "zone": ["Z1", "Z1"],
            "demand": [100.0, 101.5],
            "target_h1": [102.0, 103.0],
        }
    )

    expected.to_parquet(dataset_path, engine="pyarrow", compression=compression)
    actual = pd.read_parquet(dataset_path, engine="pyarrow")

    pdt.assert_frame_equal(actual, expected)


def test_pyarrow_roundtrip_supports_parquet_files(tmp_path: Path) -> None:
    _roundtrip_parquet(tmp_path / "dataset.parquet", compression=None)


def test_pyarrow_roundtrip_supports_gzipped_parquet_filenames(tmp_path: Path) -> None:
    _roundtrip_parquet(tmp_path / "dataset.parquet.gz", compression="gzip")
