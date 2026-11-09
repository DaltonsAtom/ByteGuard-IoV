
"""
ByteGuard-IoV — Script 07: Chronological Dataset Preparation

Objective:
    Create leakage-aware chronological train, validation,
    and test datasets from the four original CAN recordings.

Input:
    data/processed/DoS_dataset_processed.csv
    data/processed/Fuzzy_dataset_processed.csv
    data/processed/gear_dataset_processed.csv
    data/processed/RPM_dataset_processed.csv

Output:
    data/temporal/train.csv
    data/temporal/validation.csv
    data/temporal/test.csv
    data/temporal/vocab.json
    data/temporal/temporal_split_report.json

Split:
    First 70% of each recording  -> Training
    Next 15%                    -> Validation
    Final 15%                   -> Testing

Sampling:
    10,000 records per class for training.
    1,500 records per class for validation.
    1,500 records per class for testing.

Vocabulary:
    Constructed exclusively from training records.

Note:
    Sampling is performed after chronological partitioning.
    The model is NOT trained in this script.
"""

import json
import random

from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "temporal"

INPUT_FILES = [
    "DoS_dataset_processed.csv",
    "Fuzzy_dataset_processed.csv",
    "gear_dataset_processed.csv",
    "RPM_dataset_processed.csv",
]

CLASS_NAMES = {
    0: "Normal",
    1: "DoS",
    2: "Fuzzy",
    3: "Gear Spoofing",
    4: "RPM Spoofing",
}

SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

TARGET_SAMPLES = {
    "train": 10000,
    "validation": 1500,
    "test": 1500,
}

CHUNK_SIZE = 200000

# Retain more candidates from each chunk to ensure
# sufficient records for final balanced sampling.
CHUNK_SAMPLE_LIMIT = 2000

SEED = 42

random.seed(SEED)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REQUIRED_COLUMNS = [
    "Timestamp",
    "Flag",
    "Class_Label",
    "Byte_Sentence",
]


# --------------------------------------------------
# 2. Verify input files
# --------------------------------------------------

print("=" * 70)
print("BYTEGUARD-IoV — CHRONOLOGICAL DATA PREPARATION")
print("=" * 70)

for filename in INPUT_FILES:

    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}"
        )

print("All four processed datasets found.")


# --------------------------------------------------
# 3. Determine timestamp boundaries
# --------------------------------------------------

def get_timestamp_range(file_path):
    """
    Scan a recording in chunks to determine its
    minimum and maximum timestamps.
    """

    minimum = float("inf")
    maximum = float("-inf")

    total_rows = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=["Timestamp"],
        chunksize=CHUNK_SIZE,
    ):

        timestamps = pd.to_numeric(
            chunk["Timestamp"],
            errors="coerce",
        ).dropna()

        if timestamps.empty:
            continue

        minimum = min(
            minimum,
            timestamps.min(),
        )

        maximum = max(
            maximum,
            timestamps.max(),
        )

        total_rows += len(chunk)

    if total_rows == 0:
        raise ValueError(
            f"No valid timestamps in {file_path.name}"
        )

    return {
        "minimum": float(minimum),
        "maximum": float(maximum),
        "rows": total_rows,
    }


recording_ranges = {}

print("\nSCANNING RECORDING TIMESTAMPS")

for filename in INPUT_FILES:

    file_path = PROCESSED_DIR / filename

    info = get_timestamp_range(file_path)

    minimum = info["minimum"]
    maximum = info["maximum"]

    duration = maximum - minimum

    # Calculate boundaries separately for each recording.
    train_end = minimum + (
        duration * SPLIT_RATIOS["train"]
    )

    validation_end = minimum + (
        duration * (
            SPLIT_RATIOS["train"]
            + SPLIT_RATIOS["validation"]
        )
    )

    recording_ranges[filename] = {
        "minimum": minimum,
        "maximum": maximum,
        "train_end": train_end,
        "validation_end": validation_end,
        "rows": info["rows"],
    }

    print(f"\n{filename}")
    print(f"  Rows: {info['rows']:,}")
    print(f"  Minimum timestamp: {minimum:.6f}")
    print(f"  Maximum timestamp: {maximum:.6f}")
    print(f"  Training ends: {train_end:.6f}")
    print(f"  Validation ends: {validation_end:.6f}")


# --------------------------------------------------
# 4. Assign records to chronological partitions
# --------------------------------------------------

def assign_split(timestamp, boundaries):
    """
    Assign a record based on its timestamp.

    Each recording has its own chronological boundaries.
    """

    if timestamp <= boundaries["train_end"]:
        return "train"

    if timestamp <= boundaries["validation_end"]:
        return "validation"

    return "test"


# --------------------------------------------------
# 5. Collect candidate samples in chunks
# --------------------------------------------------

candidates = {
    split: {
        class_id: []
        for class_id in CLASS_NAMES
    }
    for split in SPLIT_RATIOS
}

print("\nCOLLECTING CHRONOLOGICAL SAMPLES")

for filename in INPUT_FILES:

    file_path = PROCESSED_DIR / filename

    boundaries = recording_ranges[filename]

    print(f"\nProcessing: {filename}")

    chunk_number = 0

    for chunk in pd.read_csv(
        file_path,
        usecols=REQUIRED_COLUMNS,
        chunksize=CHUNK_SIZE,
    ):

        chunk_number += 1

        # Convert and validate timestamps.
        chunk["Timestamp"] = pd.to_numeric(
            chunk["Timestamp"],
            errors="coerce",
        )

        chunk["Class_Label"] = pd.to_numeric(
            chunk["Class_Label"],
            errors="coerce",
        )

        chunk = chunk.dropna(
            subset=[
                "Timestamp",
                "Class_Label",
                "Flag",
                "Byte_Sentence",
            ]
        ).copy()

        chunk["Class_Label"] = (
            chunk["Class_Label"].astype(int)
        )

        chunk["Flag"] = (
            chunk["Flag"].astype(str).str.strip()
        )

        # Validate labels.
        chunk = chunk[
            chunk["Class_Label"].isin(CLASS_NAMES)
        ]

        # Retain only records with valid labels.
        chunk = chunk[
            (
                (chunk["Flag"] == "R")
                & (chunk["Class_Label"] == 0)
            )
            |
            (
                (chunk["Flag"] == "T")
                & (chunk["Class_Label"] != 0)
            )
        ]

        if chunk.empty:
            continue

        # Assign chronological partitions.
        chunk["Split"] = chunk["Timestamp"].apply(
            lambda timestamp: assign_split(
                timestamp,
                boundaries,
            )
        )

        # Sample candidates separately for each split and class.
        for split in SPLIT_RATIOS:

            split_chunk = chunk[
                chunk["Split"] == split
            ]

            for class_id in CLASS_NAMES:

                class_chunk = split_chunk[
                    split_chunk["Class_Label"] == class_id
                ]

                if class_chunk.empty:
                    continue

                sample_count = min(
                    len(class_chunk),
                    CHUNK_SAMPLE_LIMIT,
                )

                sampled = class_chunk.sample(
                    n=sample_count,
                    random_state=SEED + chunk_number,
                )

                candidates[split][class_id].append(
                    sampled
                )

        if chunk_number % 10 == 0:
            print(
                f"  Processed chunks: {chunk_number}"
            )


# --------------------------------------------------
# 6. Perform final balanced sampling
# --------------------------------------------------

print("\nFINAL BALANCED SAMPLING")

final_datasets = {}

for split in SPLIT_RATIOS:

    split_frames = []

    target = TARGET_SAMPLES[split]

    print(f"\n{split.upper()}")

    for class_id, class_name in CLASS_NAMES.items():

        available_frames = candidates[split][class_id]

        if not available_frames:
            raise ValueError(
                f"No candidates found for {class_name} "
                f"in {split} partition."
            )

        combined = pd.concat(
            available_frames,
            ignore_index=True,
        )

        # Remove accidental duplicate candidates.
        combined = combined.drop_duplicates(
            subset=["Timestamp", "Byte_Sentence"]
        )

        available = len(combined)

        if available < target:
            raise ValueError(
                f"Insufficient candidate records for {class_name} "
                f"in {split}: found {available:,}, "
                f"required {target:,}. "
                "Check CHUNK_SAMPLE_LIMIT and timestamp distribution."
            )

        selected = combined.sample(
            n=target,
            random_state=SEED,
        ).copy()

        selected["Class_Label"] = class_id

        split_frames.append(selected)

        print(
            f"  {class_name:18s}: "
            f"{len(selected):,} selected "
            f"from {available:,} candidates"
        )

    final_df = pd.concat(
        split_frames,
        ignore_index=True,
    )

    # Randomize record order within each partition.
    # This does not change the chronological separation.
    final_df = final_df.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    final_datasets[split] = final_df


# --------------------------------------------------
# 7. Save chronological datasets
# --------------------------------------------------

print("\nSAVING DATASETS")

for split, dataframe in final_datasets.items():

    output_file = OUTPUT_DIR / f"{split}.csv"

    dataframe.to_csv(
        output_file,
        index=False,
    )

    print(
        f"{split:12s}: "
        f"{len(dataframe):,} records "
        f"-> {output_file}"
    )


# --------------------------------------------------
# 8. Construct vocabulary from training only
# --------------------------------------------------

print("\nBUILDING TRAINING-ONLY VOCABULARY")

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
]

vocab = {
    token: index
    for index, token in enumerate(SPECIAL_TOKENS)
}

for sentence in final_datasets["train"]["Byte_Sentence"]:

    for token in str(sentence).upper().split():

        if token not in vocab:
            vocab[token] = len(vocab)

vocab_file = OUTPUT_DIR / "vocab.json"

with vocab_file.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        vocab,
        file,
        indent=4,
    )

print(f"Vocabulary size: {len(vocab):,}")
print(f"Vocabulary saved: {vocab_file}")


# --------------------------------------------------
# 9. Verify class distribution
# --------------------------------------------------

print("\nVERIFYING CLASS DISTRIBUTION")

distribution = {}

for split, dataframe in final_datasets.items():

    counts = dataframe["Class_Label"].value_counts()

    distribution[split] = {}

    for class_id, class_name in CLASS_NAMES.items():

        count = int(counts.get(class_id, 0))

        distribution[split][class_name] = count

        if count != TARGET_SAMPLES[split]:
            raise RuntimeError(
                f"Incorrect class count for {class_name} "
                f"in {split}: {count}"
            )

        print(
            f"{split:12s} | "
            f"{class_name:18s} | {count:,}"
        )


# --------------------------------------------------
# 10. Verify chronological separation
# --------------------------------------------------

print("\nVERIFYING CHRONOLOGICAL SEPARATION")

temporal_checks = {}

for filename in INPUT_FILES:

    boundaries = recording_ranges[filename]

    temporal_checks[filename] = {
        "training_end": boundaries["train_end"],
        "validation_end": boundaries["validation_end"],
        "valid": (
            boundaries["minimum"]
            < boundaries["train_end"]
            < boundaries["validation_end"]
            < boundaries["maximum"]
        ),
    }

    if not temporal_checks[filename]["valid"]:
        raise RuntimeError(
            f"Invalid temporal boundaries for {filename}"
        )

    print(f"{filename}: PASS")


# --------------------------------------------------
# 11. Generate experimental report
# --------------------------------------------------

report = {
    "experiment": "Chronological Generalisation Preparation",

    "random_seed": SEED,

    "split_ratios": SPLIT_RATIOS,

    "target_samples_per_class": TARGET_SAMPLES,

    "total_samples": sum(
        len(dataframe)
        for dataframe in final_datasets.values()
    ),

    "class_distribution": distribution,

    "vocabulary_size": len(vocab),

    "vocabulary_source": "Training partition only",

    "recording_boundaries": recording_ranges,

    "temporal_validation": temporal_checks,

    "input_files": INPUT_FILES,

    "output_directory": str(OUTPUT_DIR),
}

report_file = (
    OUTPUT_DIR / "temporal_split_report.json"
)

with report_file.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        report,
        file,
        indent=4,
    )


# --------------------------------------------------
# 12. Final summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("CHRONOLOGICAL DATA PREPARATION COMPLETE")
print("=" * 70)

for split, dataframe in final_datasets.items():

    print(
        f"{split.upper():12s}: "
        f"{len(dataframe):,} records"
    )

print(f"\nTotal records: {report['total_samples']:,}")

print(f"Vocabulary size: {len(vocab):,}")

print(f"Report: {report_file}")

print("\nAll chronological validation checks passed.")
