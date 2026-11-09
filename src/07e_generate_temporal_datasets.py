
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "temporal"

PLAN_PATH = (
    ROOT / "outputs" / "temporal_analysis" /
    "temporal_partition_plan.json"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 200_000

# Maximum candidate records retained per class per chunk.
# Final datasets are sampled from these candidates.
CHUNK_SAMPLE_LIMIT = 1500

RANDOM_SEED = 42

# Final balanced dataset sizes
TARGET_SIZES = {
    "train": 10_000,
    "validation": 1_500,
    "test": 1_500,
}

# Class encoding
CLASS_NAMES = {
    0: "Normal",
    1: "DoS",
    2: "Fuzzy",
    3: "Gear Spoofing",
    4: "RPM Spoofing",
}

FILES = {
    "DoS_dataset_processed.csv": {
        "name": "DoS",
        "class_id": 1,
    },
    "Fuzzy_dataset_processed.csv": {
        "name": "Fuzzy",
        "class_id": 2,
    },
    "gear_dataset_processed.csv": {
        "name": "Gear Spoofing",
        "class_id": 3,
    },
    "RPM_dataset_processed.csv": {
        "name": "RPM Spoofing",
        "class_id": 4,
    },
}

PARTITIONS = ["train", "validation", "test"]

REQUIRED_COLUMNS = [
    "Timestamp",
    "Flag",
    "Class_Label",
    "Byte_Sentence",
]


# ============================================================
# 2. LOAD AND VALIDATE PARTITION PLAN
# ============================================================

def load_partition_plan():

    if not PLAN_PATH.exists():
        raise FileNotFoundError(
            f"Partition plan not found: {PLAN_PATH}"
        )

    with open(PLAN_PATH, "r", encoding="utf-8") as file:
        plan = json.load(file)

    plan_lookup = {}

    for row in plan:

        dataset = row["Dataset"]
        partition = row["Partition"]

        if partition not in PARTITIONS:
            continue

        plan_lookup[(dataset, partition)] = {
            "start_window": int(row["Start_Window"]),
            "end_window": int(row["End_Window"]),
            "start_timestamp": float(row["Start_Timestamp"]),
            "end_timestamp": float(row["End_Timestamp"]),
        }

    for dataset_config in FILES.values():

        dataset = dataset_config["name"]

        for partition in PARTITIONS:

            if (dataset, partition) not in plan_lookup:
                raise ValueError(
                    f"Missing {partition} plan for {dataset}"
                )

    return plan_lookup


# ============================================================
# 3. INITIALISE CANDIDATE STORAGE
# ============================================================

def initialise_candidate_storage():

    candidates = {
        partition: {
            class_id: []
            for class_id in CLASS_NAMES
        }
        for partition in PARTITIONS
    }

    # Separate storage for later normal-only traffic.
    followup_candidates = {
        dataset: []
        for dataset in FILES
    }

    return candidates, followup_candidates


# ============================================================
# 4. SAMPLE RECORDS FROM ONE CHUNK
# ============================================================

def sample_chunk_records(
    subset,
    class_id,
    rng,
):

    if subset.empty:
        return None

    # Keep a limited number of candidates from each chunk.
    sample_count = min(
        len(subset),
        CHUNK_SAMPLE_LIMIT
    )

    sampled = subset.sample(
        n=sample_count,
        random_state=int(rng.integers(0, 2**31 - 1))
    )

    sampled = sampled.copy()

    sampled["Class_Label"] = class_id

    return sampled


# ============================================================
# 5. PROCESS ONE ORIGINAL DATASET
# ============================================================

def process_dataset(
    filename,
    config,
    plan_lookup,
    candidates,
    followup_candidates,
    rng,
):

    path = DATA_DIR / filename

    dataset_name = config["name"]
    attack_class_id = config["class_id"]

    print("\n" + "=" * 75)
    print(f"PROCESSING: {dataset_name}")
    print("=" * 75)

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}"
        )

    # The beginning of the training partition is the
    # original minimum timestamp for this recording.
    minimum_timestamp = plan_lookup[
        (dataset_name, "train")
    ]["start_timestamp"]

    # Obtain the chronological boundaries.
    boundaries = {
        partition: plan_lookup[
            (dataset_name, partition)
        ]
        for partition in PARTITIONS
    }

    # End of the final attack-containing window.
    last_attack_window = boundaries["test"]["end_window"]

    # --------------------------------------------------------
    # Read original data in chunks
    # --------------------------------------------------------

    processed_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=REQUIRED_COLUMNS,
            chunksize=CHUNK_SIZE,
        ),
        start=1
    ):

        # Normalize data types.
        chunk["Timestamp"] = pd.to_numeric(
            chunk["Timestamp"],
            errors="coerce"
        )

        chunk["Class_Label"] = pd.to_numeric(
            chunk["Class_Label"],
            errors="coerce"
        )

        chunk = chunk.dropna(
            subset=REQUIRED_COLUMNS
        )

        chunk["Class_Label"] = (
            chunk["Class_Label"].astype(int)
        )

        # Verify labels against the original Flag.
        valid_normal = (
            (chunk["Flag"] == "R") &
            (chunk["Class_Label"] == 0)
        )

        valid_attack = (
            (chunk["Flag"] == "T") &
            (chunk["Class_Label"] == attack_class_id)
        )

        valid_rows = valid_normal | valid_attack

        if not valid_rows.all():
            raise ValueError(
                f"Invalid labels detected in {dataset_name}, "
                f"chunk {chunk_number}."
            )

        # Assign source recording.
        chunk["Source_Dataset"] = dataset_name

        # Calculate the same 5-second window index used
        # during Script 07c.
        chunk["Window_ID"] = (
            (
                chunk["Timestamp"] - minimum_timestamp
            ) // 5
        ).astype(int)

        processed_rows += len(chunk)

        # ----------------------------------------------------
        # Assign records to chronological partitions
        # ----------------------------------------------------

        for partition in PARTITIONS:

            boundary = boundaries[partition]

            mask = (
                (chunk["Window_ID"] >= boundary["start_window"]) &
                (chunk["Window_ID"] <= boundary["end_window"])
            )

            partition_chunk = chunk.loc[mask]

            if partition_chunk.empty:
                continue

            # Sample normal records.
            normal_rows = partition_chunk[
                partition_chunk["Class_Label"] == 0
            ]

            normal_sample = sample_chunk_records(
                normal_rows,
                0,
                rng
            )

            if normal_sample is not None:
                candidates[partition][0].append(
                    normal_sample
                )

            # Sample attack records.
            attack_rows = partition_chunk[
                partition_chunk["Class_Label"] == attack_class_id
            ]

            attack_sample = sample_chunk_records(
                attack_rows,
                attack_class_id,
                rng
            )

            if attack_sample is not None:
                candidates[partition][attack_class_id].append(
                    attack_sample
                )

        # ----------------------------------------------------
        # Collect post-attack normal-only traffic
        # ----------------------------------------------------

        followup_mask = (
            (chunk["Window_ID"] > last_attack_window) &
            (chunk["Class_Label"] == 0)
        )

        followup_rows = chunk.loc[followup_mask]

        if not followup_rows.empty:

            followup_sample = followup_rows.sample(
                n=min(
                    len(followup_rows),
                    CHUNK_SAMPLE_LIMIT
                ),
                random_state=int(
                    rng.integers(0, 2**31 - 1)
                )
            )

            followup_candidates[filename].append(
                followup_sample.copy()
            )

        if chunk_number % 10 == 0:
            print(
                f"Processed chunks: {chunk_number} | "
                f"Rows scanned: {processed_rows:,}"
            )

    print(
        f"Total rows scanned: {processed_rows:,}"
    )


# ============================================================
# 6. BUILD FINAL BALANCED PARTITION
# ============================================================

def build_partition(
    partition,
    candidates,
    rng,
):

    target = TARGET_SIZES[partition]

    print("\n" + "-" * 75)
    print(f"BUILDING {partition.upper()} DATASET")
    print("-" * 75)

    selected_classes = []

    for class_id, class_name in CLASS_NAMES.items():

        frames = candidates[partition][class_id]

        if not frames:
            raise ValueError(
                f"No candidates found for {class_name} "
                f"in {partition}."
            )

        combined = pd.concat(
            frames,
            ignore_index=True
        )

        # Remove accidental duplicates using source,
        # timestamp and byte sentence.
        combined = combined.drop_duplicates(
            subset=[
                "Source_Dataset",
                "Timestamp",
                "Byte_Sentence",
            ]
        )

        available = len(combined)

        if available < target:
            raise ValueError(
                f"Insufficient {class_name} candidates in "
                f"{partition}: {available:,} available, "
                f"{target:,} required."
            )

        selected = combined.sample(
            n=target,
            random_state=int(
                rng.integers(0, 2**31 - 1)
            )
        ).copy()

        selected["Class_Name"] = class_name

        selected_classes.append(selected)

        print(
            f"{class_name:18s}: "
            f"{target:,} selected from {available:,} candidates"
        )

    result = pd.concat(
        selected_classes,
        ignore_index=True
    )

    # Keep each recording chronological.
    result = result.sort_values(
        ["Source_Dataset", "Timestamp"]
    ).reset_index(drop=True)

    # Remove internal window index from model input.
    result = result.drop(
        columns=["Window_ID"],
        errors="ignore"
    )

    return result


# ============================================================
# 7. BUILD POST-ATTACK NORMAL-ONLY DATASET
# ============================================================

def build_followup_dataset(
    followup_candidates,
):

    print("\n" + "=" * 75)
    print("BUILDING POST-ATTACK NORMAL-ONLY DATASET")
    print("=" * 75)

    frames = []

    # Retain up to 5,000 normal records from each recording.
    target_per_dataset = 5000

    for filename, config in FILES.items():

        dataset_name = config["name"]

        stored_frames = followup_candidates[filename]

        if not stored_frames:
            print(
                f"{dataset_name}: No follow-up candidates."
            )
            continue

        combined = pd.concat(
            stored_frames,
            ignore_index=True
        )

        combined = combined.drop_duplicates(
            subset=[
                "Source_Dataset",
                "Timestamp",
                "Byte_Sentence",
            ]
        )

        selected_count = min(
            target_per_dataset,
            len(combined)
        )

        selected = combined.sample(
            n=selected_count,
            random_state=RANDOM_SEED
        ).copy()

        selected["Class_Name"] = "Normal"

        frames.append(selected)

        print(
            f"{dataset_name:18s}: "
            f"{selected_count:,} normal records"
        )

    if not frames:
        raise ValueError(
            "No post-attack normal records were collected."
        )

    result = pd.concat(
        frames,
        ignore_index=True
    )

    result = result.sort_values(
        ["Source_Dataset", "Timestamp"]
    ).reset_index(drop=True)

    result = result.drop(
        columns=["Window_ID"],
        errors="ignore"
    )

    return result


# ============================================================
# 8. GENERATE DATASET REPORT
# ============================================================

def generate_report(datasets):

    report = {}

    for partition, df in datasets.items():

        counts = (
            df["Class_Name"]
            .value_counts()
            .to_dict()
        )

        report[partition] = {
            "total_records": len(df),
            "class_distribution": counts,
            "source_distribution": (
                df["Source_Dataset"]
                .value_counts()
                .to_dict()
            ),
        }

    return report


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 80)
    print("BYTEGUARD-IoV: TEMPORAL DATASET GENERATION")
    print("=" * 80)

    rng = np.random.default_rng(RANDOM_SEED)

    plan_lookup = load_partition_plan()

    candidates, followup_candidates = (
        initialise_candidate_storage()
    )

    # Process all four original recordings.
    for filename, config in FILES.items():

        process_dataset(
            filename=filename,
            config=config,
            plan_lookup=plan_lookup,
            candidates=candidates,
            followup_candidates=followup_candidates,
            rng=rng,
        )

    # Generate final balanced datasets.
    datasets = {}

    for partition in PARTITIONS:

        datasets[partition] = build_partition(
            partition=partition,
            candidates=candidates,
            rng=rng,
        )

    # Generate separate normal-only follow-up dataset.
    datasets["normal_followup"] = build_followup_dataset(
        followup_candidates
    )

    # --------------------------------------------------------
    # Save final CSV files
    # --------------------------------------------------------

    for partition, df in datasets.items():

        output_path = (
            OUTPUT_DIR / f"{partition}.csv"
        )

        df.to_csv(
            output_path,
            index=False
        )

        print(
            f"\nSaved {partition}: {output_path}"
        )

    # --------------------------------------------------------
    # Save dataset report
    # --------------------------------------------------------

    report = generate_report(datasets)

    report_path = (
        OUTPUT_DIR / "temporal_dataset_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\n" + "=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)

    for partition, details in report.items():

        print(f"\n{partition.upper()}")

        print(
            f"Total records: {details['total_records']:,}"
        )

        for class_name, count in (
            details["class_distribution"].items()
        ):
            print(f"  {class_name:18s}: {count:,}")

    print("\nReport saved:")
    print(report_path)

    print("\n" + "=" * 80)
    print("TEMPORAL DATASET GENERATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
