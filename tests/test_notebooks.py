# pylint: disable=missing-module-docstring, missing-function-docstring

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOKS = [
    "00_setup.ipynb",
    "01_baselines.ipynb",
    "02_xgboost.ipynb",
    "03_multi_horizon.ipynb",
    "04_tft_inference.ipynb",
]


def test_notebooks_are_output_free_and_metadata_clean() -> None:
    notebook_dir = Path("notebooks")
    for notebook_name in NOTEBOOKS:
        notebook = json.loads(
            (notebook_dir / notebook_name).read_text(encoding="utf-8")
        )
        assert notebook["metadata"].get("language_info", {}).get("name") == "python"
        for cell in notebook["cells"]:
            assert cell.get("outputs", []) == []
            assert "execution_count" not in cell or cell["execution_count"] is None
