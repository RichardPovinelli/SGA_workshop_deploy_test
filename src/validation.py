"""Dataset validation and data leakage detection."""

from __future__ import annotations

import pandas as pd


def check_data_leakage(df: pd.DataFrame) -> dict[str, bool]:
    """Verify no data leakage between train and test sets.

    Checks:
    - No overlapping indices between train and test
    - Temporal ordering (all train rows before test rows per zone)
    - Exactly 365 test rows per zone
    - No NaN values in features used for modeling

    Returns:
        Dictionary with zone-level and overall leakage status.
    """
    results = {}
    all_passed = True

    for zone in sorted(df["zone"].unique()):
        zone_df = df[df["zone"] == zone].sort_index()
        train_indices = zone_df[zone_df["split"] == "train"].index
        test_indices = zone_df[zone_df["split"] == "test"].index

        zone_passed = True

        # Check for overlapping indices
        overlap = set(train_indices) & set(test_indices)
        if overlap:
            print(f"  {zone}: ERROR - Overlapping indices: {overlap}")
            zone_passed = False
        else:
            print(f"  {zone}: ✓ No overlapping indices")

        # Check temporal ordering (train should come before test)
        if len(train_indices) > 0 and len(test_indices) > 0:
            max_train = train_indices.max()
            min_test = test_indices.min()
            if max_train < min_test:
                print(f"  {zone}: ✓ Temporal order correct (train < test)")
            else:
                print(f"  {zone}: ERROR - Temporal leakage detected!")
                zone_passed = False

        # Verify exact count (365 test rows per zone)
        if len(test_indices) == 365:
            print(f"  {zone}: ✓ Exactly 365 test rows")
        else:
            print(f"  {zone}: Warning - {len(test_indices)} test rows (expected 365)")
            zone_passed = False

        results[zone] = zone_passed
        if not zone_passed:
            all_passed = False

    # Check no NaN in features used for modeling
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    feature_cols = [col for col in df.columns if col not in ["date", "zone", "split"]]
    train_nans = train_df[feature_cols].isnull().sum().sum()
    test_nans = test_df[feature_cols].isnull().sum().sum()

    print(f"\nFeature NaN check (all zones):")
    print(f"  Train set: {train_nans} NaN values")
    print(f"  Test set:  {test_nans} NaN values")

    if train_nans > 0 or test_nans > 0:
        all_passed = False

    results["overall"] = all_passed
    return results


if __name__ == "__main__":
    from src.data.load import load_dataset

    print("DATA LEAKAGE VERIFICATION\n")
    df = load_dataset()
    check_data_leakage(df)
    print("\n✓ Data leakage verification complete")
