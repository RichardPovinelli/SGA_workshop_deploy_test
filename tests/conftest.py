# pylint: disable=missing-module-docstring, missing-function-docstring

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_model_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    zones = ("zone_a", "zone_b")

    for zone_index, zone in enumerate(zones):
        zone_offset = float(zone_index * 3)
        for day in range(10):
            demand = 100.0 + zone_offset + day
            temp = 20.0 + (day % 3)
            hdd = max(0.0, 12.0 - temp)
            cdd = max(0.0, temp - 24.0)
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01")
                    + pd.Timedelta(days=day + zone_index * 20),
                    "zone": zone,
                    "split": "train" if day < 7 else "test",
                    "demand": demand,
                    "temp": temp,
                    "hdd": hdd,
                    "cdd": cdd,
                    "dow": float((day % 7) + 1),
                    "is_weekend": float(1 if day % 7 in (5, 6) else 0),
                    "month": 1.0,
                    "lag1": demand - 1.0,
                    "lag7": demand - 7.0,
                    "target_h1": demand + 0.5,
                    "target_h2": demand + 1.0,
                    "target_h3": demand + 1.5,
                    "target_h4": demand + 2.0,
                    "target_h5": demand + 2.5,
                    "target_h6": demand + 3.0,
                    "target_h7": demand + 3.5,
                }
            )

    return pd.DataFrame(rows)
