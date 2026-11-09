
import pandas as pd
from pathlib import Path

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

CHUNK_SIZE = 200_000

# Dataset filenames and their expected attack class IDs
FILES = {
    "DoS_dataset_processed.csv": {
        "name": "DoS",
        "class_id": 1
    },
    "Fuzzy_dataset_processed.csv": {
        "name": "Fuzzy",
        "class_id": 2
    },
    "gear_dataset_processed.csv": {
        "name": "Gear Spoofing",
        "class_id": 3
    },
    "RPM_dataset_processed.csv": {
        "name": "RPM Spoofing",
        "class_id": 4
    }
}


# ============================================================
# 2. DATASET INSPECTION FUNCTION
# ============================================================

def inspect_dataset(filename, attack_name, attack_class_id):

    path = DATA_DIR / filename

    if not path.exists():
        print(f"\nERROR: File not found: {path}")
        return

    print("\n" + "=" * 75)
    print(f"DATASET: {attack_name}")
    print("=" * 75)

    # Timestamp boundaries
    attack_min = float("inf")
    attack_max = float("-inf")

    normal_min = float("inf")
    normal_max = float("-inf")

    # Record counts
    attack_count = 0
    normal_count = 0

    # Label validation
    mismatches = 0

    # Process the dataset in manageable chunks
    for chunk in pd.read_csv(
        path,
        usecols=["Timestamp", "Flag", "Class_Label"],
        chunksize=CHUNK_SIZE
    ):

        chunk["Timestamp"] = pd.to_numeric(
            chunk["Timestamp"],
            errors="coerce"
        )

        chunk["Class_Label"] = pd.to_numeric(
            chunk["Class_Label"],
            errors="coerce"
        )

        # Remove rows without the required fields
        chunk = chunk.dropna(
            subset=["Timestamp", "Flag", "Class_Label"]
        )

        # Separate normal and attack records
        attack = chunk[chunk["Flag"] == "T"]
        normal = chunk[chunk["Flag"] == "R"]

        # ----------------------------------------------------
        # Attack statistics
        # ----------------------------------------------------

        if not attack.empty:

            attack_count += len(attack)

            attack_min = min(
                attack_min,
                attack["Timestamp"].min()
            )

            attack_max = max(
                attack_max,
                attack["Timestamp"].max()
            )

        # ----------------------------------------------------
        # Normal statistics
        # ----------------------------------------------------

        if not normal.empty:

            normal_count += len(normal)

            normal_min = min(
                normal_min,
                normal["Timestamp"].min()
            )

            normal_max = max(
                normal_max,
                normal["Timestamp"].max()
            )

        # ----------------------------------------------------
        # Validate Flag and Class_Label consistency
        #
        # Flag T -> expected attack class ID
        # Flag R -> expected class ID 0
        # ----------------------------------------------------

        attack_mismatches = (
            (chunk["Flag"] == "T") &
            (chunk["Class_Label"] != attack_class_id)
        )

        normal_mismatches = (
            (chunk["Flag"] == "R") &
            (chunk["Class_Label"] != 0)
        )

        mismatches += int(
            attack_mismatches.sum() +
            normal_mismatches.sum()
        )

        # Check for unexpected Flag values
        invalid_flags = ~chunk["Flag"].isin(["T", "R"])

        mismatches += int(invalid_flags.sum())

    # ========================================================
    # 3. DISPLAY RESULTS
    # ========================================================

    print(f"\nNormal records : {normal_count:,}")
    print(f"Attack records : {attack_count:,}")

    if attack_count > 0:

        print("\nATTACK TIMESTAMP INTERVAL")

        print(f"Start: {attack_min:.6f}")
        print(f"End  : {attack_max:.6f}")

    if normal_count > 0:

        print("\nNORMAL TIMESTAMP INTERVAL")

        print(f"Start: {normal_min:.6f}")
        print(f"End  : {normal_max:.6f}")

    print("\nLABEL VALIDATION")

    print(f"Expected attack class ID: {attack_class_id}")
    print("Expected normal class ID: 0")
    print(f"Flag/Class_Label mismatches: {mismatches:,}")

    if mismatches == 0:
        print("STATUS: PASS - Labels are consistent.")
    else:
        print("STATUS: WARNING - Label inconsistencies detected.")


# ============================================================
# 4. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 75)
    print("BYTEGUARD-IoV: ATTACK INTERVAL AND LABEL INSPECTION")
    print("=" * 75)

    for filename, config in FILES.items():

        inspect_dataset(
            filename=filename,
            attack_name=config["name"],
            attack_class_id=config["class_id"]
        )

    print("\n" + "=" * 75)
    print("INSPECTION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
