
"""
ByteGuard-IoV — Script 02: Dataset Preprocessing

Purpose:
    Parse variable-length CAN records correctly, validate them,
    and assign binary and multiclass intrusion labels.

Inputs:
    data/raw/DoS_dataset.csv
    data/raw/Fuzzy_dataset.csv
    data/raw/gear_dataset.csv
    data/raw/RPM_dataset.csv

Outputs:
    data/processed/DoS_processed.csv
    data/processed/Fuzzy_processed.csv
    data/processed/gear_processed.csv
    data/processed/RPM_processed.csv
    data/processed/preprocessing_report.json

Important:
    The original CSV files are never modified.
    DLC determines the number of payload bytes.
    The Flag follows the actual payload, not necessarily DATA[7].
"""

import csv
import json
from collections import Counter
from pathlib import Path


# --------------------------------------------------
# 1. Project paths and dataset configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASETS = {
    "DoS_dataset.csv": "DoS",
    "Fuzzy_dataset.csv": "Fuzzy",
    "gear_dataset.csv": "Gear Spoofing",
    "RPM_dataset.csv": "RPM Spoofing",
}

PAYLOAD_COLUMNS = [f"DATA[{i}]" for i in range(8)]

OUTPUT_COLUMNS = [
    "Timestamp",
    "CAN_ID",
    "DLC",
    *PAYLOAD_COLUMNS,
    "Flag",
    "Attack_Type",
    "Binary_Label",
    "Class_Label",
    "Byte_Sentence",
]

# Binary encoding: Normal = 0, Attack = 1.
BINARY_LABELS = {
    "R": 0,
    "T": 1,
}

# Five-class encoding.
CLASS_LABELS = {
    "Normal": 0,
    "DoS": 1,
    "Fuzzy": 2,
    "Gear Spoofing": 3,
    "RPM Spoofing": 4,
}


# --------------------------------------------------
# 2. Validate individual CAN fields
# --------------------------------------------------

def is_hex(value: str) -> bool:
    """Return True if a non-empty string contains only hex digits."""
    if not value:
        return False

    try:
        int(value, 16)
        return True
    except ValueError:
        return False


def parse_record(row, attack_name):
    """
    Parse one variable-length CSV record.

    Expected structure:
        Timestamp, CAN_ID, DLC, DATA[0] ... DATA[DLC-1], Flag

    Returns:
        A standardised output record, or None if invalid.
        A reason string explains why a record was rejected.
    """

    # Remove surrounding whitespace and any empty trailing fields.
    row = [field.strip() for field in row]

    # The first three fields must exist.
    if len(row) < 4:
        return None, "too_short"

    timestamp_text, can_id, dlc_text = row[:3]

    # Validate timestamp.
    try:
        timestamp = float(timestamp_text)

        if not timestamp == timestamp:  # Reject NaN.
            return None, "invalid_timestamp"

        if timestamp in (float("inf"), float("-inf")):
            return None, "invalid_timestamp"

    except ValueError:
        return None, "invalid_timestamp"

    # Validate CAN ID.
    if not is_hex(can_id):
        return None, "invalid_can_id"

    can_id = can_id.upper()

    # Validate DLC: standard CAN data frames contain 0–8 bytes.
    try:
        dlc = int(dlc_text)
    except ValueError:
        return None, "invalid_dlc"

    if not 0 <= dlc <= 8:
        return None, "invalid_dlc"

    # The flag must appear immediately after the DLC payload bytes.
    flag_index = 3 + dlc

    if len(row) <= flag_index:
        return None, "missing_flag"

    flag = row[flag_index].upper()

    if flag not in BINARY_LABELS:
        return None, "invalid_flag"

    # Extract only the payload bytes declared by DLC.
    payload = row[3:flag_index]

    # A DLC-sized payload must contain valid hexadecimal bytes.
    for byte in payload:
        if (
            len(byte) != 2
            or not is_hex(byte)
        ):
            return None, "invalid_payload"

    payload = [byte.upper() for byte in payload]

    # Determine the semantic class from the flag and source file.
    if flag == "R":
        attack_type = "Normal"
    else:
        attack_type = attack_name

    binary_label = BINARY_LABELS[flag]
    class_label = CLASS_LABELS[attack_type]

    # Build the byte sentence used by the later Transformer stage.
    # Include the CAN ID followed by the actual payload bytes.
    byte_sentence = " ".join([can_id] + payload)

    # Standardise to eight payload columns.
    # Empty strings indicate positions beyond the actual DLC.
    padded_payload = payload + [""] * (8 - dlc)

    output_record = [
        timestamp,
        can_id,
        dlc,
        *padded_payload,
        flag,
        attack_type,
        binary_label,
        class_label,
        byte_sentence,
    ]

    return output_record, None


# --------------------------------------------------
# 3. Process one dataset using streaming I/O
# --------------------------------------------------

def process_dataset(filename, attack_name):
    """Process one dataset without retaining all rows in memory."""

    input_path = RAW_DIR / filename

    output_name = filename.replace(".csv", "_processed.csv")
    output_path = PROCESSED_DIR / output_name

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    counts = Counter()
    rejection_reasons = Counter()

    print(f"\nProcessing: {filename}")
    print(f"Attack category: {attack_name}")

    # Read and write incrementally to limit memory consumption.
    with input_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as destination:

        reader = csv.reader(source)
        writer = csv.writer(destination)

        writer.writerow(OUTPUT_COLUMNS)

        for row in reader:
            counts["input_rows"] += 1

            record, reason = parse_record(row, attack_name)

            if record is None:
                counts["rejected_rows"] += 1
                rejection_reasons[reason] += 1
                continue

            writer.writerow(record)

            counts["accepted_rows"] += 1

            flag = record[11]
            attack_type = record[12]

            counts[f"flag_{flag}"] += 1
            counts[f"class_{attack_type}"] += 1

    summary = {
        "input_file": filename,
        "output_file": output_name,
        "attack_category": attack_name,
        "input_rows": counts["input_rows"],
        "accepted_rows": counts["accepted_rows"],
        "rejected_rows": counts["rejected_rows"],
        "flag_R": counts["flag_R"],
        "flag_T": counts["flag_T"],
        "class_counts": {
            key.removeprefix("class_"): value
            for key, value in counts.items()
            if key.startswith("class_")
        },
        "rejection_reasons": dict(rejection_reasons),
    }

    print(f"Input rows:    {summary['input_rows']:,}")
    print(f"Accepted rows: {summary['accepted_rows']:,}")
    print(f"Rejected rows: {summary['rejected_rows']:,}")
    print(f"Normal (R):    {summary['flag_R']:,}")
    print(f"Injected (T):  {summary['flag_T']:,}")
    print(f"Output:        {output_path}")

    return summary


# --------------------------------------------------
# 4. Main execution
# --------------------------------------------------

def main():
    """Process all four source datasets and save a report."""

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DIR}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for filename, attack_name in DATASETS.items():
        summary = process_dataset(filename, attack_name)
        all_summaries.append(summary)

    report_path = PROCESSED_DIR / "preprocessing_report.json"

    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(all_summaries, report_file, indent=4)

    print("\n" + "=" * 65)
    print("PREPROCESSING SUMMARY")
    print("=" * 65)

    total_input = sum(item["input_rows"] for item in all_summaries)
    total_accepted = sum(item["accepted_rows"] for item in all_summaries)
    total_rejected = sum(item["rejected_rows"] for item in all_summaries)

    print(f"Total input rows:    {total_input:,}")
    print(f"Total accepted rows: {total_accepted:,}")
    print(f"Total rejected rows: {total_rejected:,}")
    print(f"Report saved to: {report_path}")

    print("\nReview the report before starting model training.")


if __name__ == "__main__":
    main()
