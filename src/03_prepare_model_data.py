
"""
ByteGuard-IoV — Script 03: Prepare Model Data

Purpose:
    Create a reproducible, balanced 50,000-record sample from the
    four processed CAN datasets and split it into train, validation,
    and test sets.

Inputs:
    data/processed/DoS_dataset_processed.csv
    data/processed/Fuzzy_dataset_processed.csv
    data/processed/gear_dataset_processed.csv
    data/processed/RPM_dataset_processed.csv

Outputs:
    data/processed/model_sample.csv
    data/processed/train.csv
    data/processed/validation.csv
    data/processed/test.csv
    data/processed/model_data_report.json

Note:
    Sampling is performed in chunks to avoid loading all 16.5 million
    records into memory. The split is stratified and reproducible.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASET_FILES = [
    "DoS_dataset_processed.csv",
    "Fuzzy_dataset_processed.csv",
    "gear_dataset_processed.csv",
    "RPM_dataset_processed.csv",
]

# Each attack category contributes 10,000 records.
ATTACK_SAMPLES_PER_FILE = 10_000

# Four datasets contribute 2,500 normal records each.
NORMAL_SAMPLES_PER_FILE = 2_500

RANDOM_SEED = 42
CHUNK_SIZE = 100_000

OUTPUT_COLUMNS = [
    "Timestamp",
    "CAN_ID",
    "DLC",
    "Flag",
    "Attack_Type",
    "Binary_Label",
    "Class_Label",
    "Byte_Sentence",
]


# --------------------------------------------------
# 2. Sample one processed dataset in chunks
# --------------------------------------------------

def sample_dataset(filename):
    """
    Sample normal records and the attack records associated
    with one source file.

    A small random sample is taken from each chunk. Candidates
    are combined and sampled again to reach the target sizes.
    """

    file_path = PROCESSED_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {file_path}\n"
            "Run Script 02 before running Script 03."
        )

    print(f"\nSampling: {filename}")

    normal_candidates = []
    attack_candidates = []

    # Read only the fields required for model training.
    for chunk in pd.read_csv(
        file_path,
        usecols=OUTPUT_COLUMNS,
        chunksize=CHUNK_SIZE,
    ):
        # Separate normal and injected messages.
        normal_chunk = chunk[chunk["Binary_Label"] == 0]
        attack_chunk = chunk[chunk["Binary_Label"] == 1]

        # Retain a bounded number of candidates from each chunk.
        # Final sampling below enforces exact sample sizes.
        if not normal_chunk.empty:
            normal_candidates.append(
                normal_chunk.sample(
                    n=min(500, len(normal_chunk)),
                    random_state=RANDOM_SEED,
                )
            )

        if not attack_chunk.empty:
            attack_candidates.append(
                attack_chunk.sample(
                    n=min(500, len(attack_chunk)),
                    random_state=RANDOM_SEED,
                )
            )

    if not normal_candidates or not attack_candidates:
        raise ValueError(
            f"Could not find both normal and attack records in {filename}."
        )

    # Combine the sampled candidates from all chunks.
    normal_pool = pd.concat(normal_candidates, ignore_index=True)
    attack_pool = pd.concat(attack_candidates, ignore_index=True)

    # Verify that enough candidates were collected.
    if len(normal_pool) < NORMAL_SAMPLES_PER_FILE:
        raise ValueError(
            f"Not enough normal candidates in {filename}: "
            f"{len(normal_pool)}"
        )

    if len(attack_pool) < ATTACK_SAMPLES_PER_FILE:
        raise ValueError(
            f"Not enough attack candidates in {filename}: "
            f"{len(attack_pool)}"
        )

    # Draw the required sample sizes reproducibly.
    normal_sample = normal_pool.sample(
        n=NORMAL_SAMPLES_PER_FILE,
        random_state=RANDOM_SEED,
    )

    attack_sample = attack_pool.sample(
        n=ATTACK_SAMPLES_PER_FILE,
        random_state=RANDOM_SEED,
    )

    # Combine normal and attack records from this scenario.
    sampled = pd.concat(
        [normal_sample, attack_sample],
        ignore_index=True,
    )

    return sampled


# --------------------------------------------------
# 3. Build the complete balanced dataset
# --------------------------------------------------

def main():
    """Sample all four datasets and create reproducible splits."""

    all_samples = []

    for filename in DATASET_FILES:
        sampled = sample_dataset(filename)
        all_samples.append(sampled)

    # Four source files contribute 12,500 records each.
    model_data = pd.concat(all_samples, ignore_index=True)

    # Shuffle the combined dataset deterministically.
    model_data = model_data.sample(
        frac=1,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    # Ensure the combined dataset contains exactly 50,000 records.
    if len(model_data) != 50_000:
        raise ValueError(
            f"Expected 50,000 records, got {len(model_data)}."
        )

    # --------------------------------------------------
    # 4. Split into training, validation, and test sets
    # --------------------------------------------------

    # First: 70% training and 30% temporary data.
    train_data, temporary_data = train_test_split(
        model_data,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=model_data["Class_Label"],
    )

    # Second: divide the temporary data equally.
    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temporary_data["Class_Label"],
    )

    # --------------------------------------------------
    # 5. Save the model datasets
    # --------------------------------------------------

    model_data.to_csv(
        PROCESSED_DIR / "model_sample.csv",
        index=False,
    )

    train_data.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False,
    )

    validation_data.to_csv(
        PROCESSED_DIR / "validation.csv",
        index=False,
    )

    test_data.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False,
    )

    # --------------------------------------------------
    # 6. Generate a verification report
    # --------------------------------------------------

    report = {
        "random_seed": RANDOM_SEED,
        "total_sampled_records": len(model_data),
        "split_sizes": {
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data),
        },
        "class_distribution": {
            str(class_id): int(count)
            for class_id, count in
            model_data["Class_Label"].value_counts().sort_index().items()
        },
        "class_names": {
            "0": "Normal",
            "1": "DoS",
            "2": "Fuzzy",
            "3": "Gear Spoofing",
            "4": "RPM Spoofing",
        },
    }

    report_path = PROCESSED_DIR / "model_data_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # --------------------------------------------------
    # 7. Print final summary
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("MODEL DATA PREPARATION COMPLETE")
    print("=" * 65)

    print(f"Total records: {len(model_data):,}")
    print(f"Training:      {len(train_data):,}")
    print(f"Validation:    {len(validation_data):,}")
    print(f"Testing:       {len(test_data):,}")

    print("\nClass distribution:")
    print(model_data["Attack_Type"].value_counts().to_string())

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
