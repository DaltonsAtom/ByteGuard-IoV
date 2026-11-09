
"""
ByteGuard-IoV — Script 01: Dataset Inspection

Purpose:
    Inspect the four Car-Hacking CAN datasets before preprocessing.

Input:
    data/raw/DoS_dataset.csv
    data/raw/Fuzzy_dataset.csv
    data/raw/gear_dataset.csv
    data/raw/RPM_dataset.csv

Output:
    Dataset summaries printed to the terminal.

Note:
    The original CSV files are never modified.
"""

from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

# Resolve the project root from this script's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATASET_FILES = [
    "DoS_dataset.csv",
    "Fuzzy_dataset.csv",
    "gear_dataset.csv",
    "RPM_dataset.csv",
]


# --------------------------------------------------
# 2. Inspect each dataset
# --------------------------------------------------

def inspect_dataset(filename: str) -> None:
    """Load and print a summary of one dataset."""

    file_path = RAW_DATA_DIR / filename

    print("\n" + "=" * 75)
    print(f"DATASET: {filename}")
    print("=" * 75)

    # Stop clearly if the expected file is missing.
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    # Load the CSV without changing its contents.
    try:
        # The source files have no header row.
        # Assign the 12 expected column names explicitly.
        column_names = [
            "Timestamp",
            "CAN_ID",
            "DLC",
            "DATA[0]",
            "DATA[1]",
            "DATA[2]",
            "DATA[3]",
            "DATA[4]",
            "DATA[5]",
            "DATA[6]",
            "DATA[7]",
            "Flag",
        ]

        df = pd.read_csv(
            file_path,
            header=None,
            names=column_names,
            low_memory=False,
        )
    except Exception as error:
        print(f"ERROR: Could not read CSV: {error}")
        return

    # Basic dataset information.
    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    # Missing-value analysis.
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    # Duplicate analysis.
    print(f"\nExact duplicate rows: {df.duplicated().sum():,}")

    # Show a small sample for understanding the format.
    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    # Identify the Flag column, ignoring case and whitespace.
    flag_column = next(
        (
            column
            for column in df.columns
            if column.strip().lower() == "flag"
        ),
        None,
    )

    if flag_column is None:
        print("\nWARNING: No 'Flag' column found.")
        print("We must inspect the actual headers before preprocessing.")
    else:
        print("\nFlag distribution:")
        print(
            df[flag_column]
            .astype("string")
            .str.strip()
            .str.upper()
            .value_counts(dropna=False)
            .to_string()
        )

    print("\nDataset inspection complete.")


# --------------------------------------------------
# 3. Run inspection on all four datasets
# --------------------------------------------------

def main() -> None:
    """Inspect every expected dataset."""

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DATA_DIR}\n"
            "Create data/raw/ inside the project root and place "
            "the four CSV files there."
        )

    for filename in DATASET_FILES:
        inspect_dataset(filename)

    print("\nAll dataset inspections have finished.")


if __name__ == "__main__":
    main()
