"""Data inspection and reporting utilities."""

import pandas as pd


def print_dataset_overview(df: pd.DataFrame) -> None:
    """Print comprehensive overview of dataset structure and composition.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to analyze
    """
    print("Model Features:")
    # Show only model feature columns (exclude metadata and targets)
    exclude_cols = {"date", "split"} | {f"target_h{i}" for i in range(1, 8)}
    cols = [col for col in df.columns if col not in exclude_cols]
    cols_per_group = (len(cols) + 2) // 3  # Ceiling division for even distribution

    for i in range(cols_per_group):
        row_parts = []
        for group in range(3):
            idx = group * cols_per_group + i
            if idx < len(cols):
                col_num = idx + 1
                row_parts.append(f"{col_num:2d}. {cols[idx]:<15}")
            else:
                row_parts.append("")
        print("  " + "  ".join(row_parts))

    print()

    # Count rows by split (if split column exists)
    if "split" in df.columns:
        splits = df["split"].value_counts()
        print("Data Split:")
        for split_name in sorted(splits.index):
            count = splits.get(split_name, 0)
            pct = 100 * count / len(df)
            print(f"  {split_name.title():10s} {count:5,d} rows ({pct:5.1f}%)")
        print()

    # Feature info
    if "demand" in df.columns:
        print(f"Demand range: [{df['demand'].min():.1f}, {df['demand'].max():.1f}]")
    if "temperature" in df.columns:
        print(f"Temperature range: [{df['temperature'].min():.1f}, {df['temperature'].max():.1f}]")


def print_demand_statistics(df: pd.DataFrame) -> None:
    """Print descriptive statistics for demand feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with demand column
    """
    if "demand" not in df.columns:
        print("No 'demand' column found in dataframe")
        return

    stats = df["demand"].describe()
    # Round all values to 4 decimal places
    stats_rounded = stats.round(4)
    print(stats_rounded)
