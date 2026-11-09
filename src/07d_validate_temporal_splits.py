
import pandas as pd
import json
from pathlib import Path
import math

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "outputs" / "temporal_analysis"
OUTPUT_DIR = ROOT / "outputs" / "temporal_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "DoS": "dos_5sec_windows.csv",
    "Fuzzy": "fuzzy_5sec_windows.csv",
    "Gear Spoofing": "gear_spoofing_5sec_windows.csv",
    "RPM Spoofing": "rpm_spoofing_5sec_windows.csv",
}

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# 2. VALIDATE CONFIGURATION
# ============================================================

if not math.isclose(
    TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO,
    1.0,
    abs_tol=1e-9
):
    raise ValueError("Split ratios must add up to 1.0.")


# ============================================================
# 3. ANALYSE ONE DATASET
# ============================================================

def validate_dataset(dataset_name, filename):

    path = INPUT_DIR / filename

    print("\n" + "=" * 85)
    print(f"DATASET: {dataset_name}")
    print("=" * 85)

    if not path.exists():
        raise FileNotFoundError(
            f"Required temporal report not found: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "Window_ID",
        "Start_Timestamp",
        "End_Timestamp",
        "Normal_Count",
        "Attack_Count",
        "Total_Count",
        "Window_Type",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {filename}: {sorted(missing)}"
        )

    # Ensure chronological ordering
    df = df.sort_values("Window_ID").reset_index(drop=True)

    # --------------------------------------------------------
    # Identify the attack-containing period
    # --------------------------------------------------------

    attack_windows = df[df["Attack_Count"] > 0]

    if attack_windows.empty:
        raise ValueError(
            f"No attack windows found in {dataset_name}."
        )

    first_attack_window = int(
        attack_windows["Window_ID"].min()
    )

    last_attack_window = int(
        attack_windows["Window_ID"].max()
    )

    # The active period begins at the first attack window.
    # It ends at the last attack-containing window.
    active = df[
        (df["Window_ID"] >= first_attack_window) &
        (df["Window_ID"] <= last_attack_window)
    ].copy()

    active = active.sort_values("Window_ID").reset_index(drop=True)

    number_of_active_windows = len(active)

    # --------------------------------------------------------
    # Calculate chronological boundaries
    # --------------------------------------------------------

    train_end_position = int(
        number_of_active_windows * TRAIN_RATIO
    )

    validation_end_position = int(
        number_of_active_windows *
        (TRAIN_RATIO + VALIDATION_RATIO)
    )

    # Ensure that every partition receives at least one window
    train_end_position = max(1, train_end_position)

    validation_end_position = max(
        train_end_position + 1,
        validation_end_position
    )

    validation_end_position = min(
        validation_end_position,
        number_of_active_windows - 1
    )

    if not (
        0 < train_end_position <
        validation_end_position <
        number_of_active_windows
    ):
        raise ValueError(
            f"Not enough active windows to split {dataset_name}."
        )

    # --------------------------------------------------------
    # Assign partitions without shuffling
    # --------------------------------------------------------

    train = active.iloc[:train_end_position]

    validation = active.iloc[
        train_end_position:validation_end_position
    ]

    test = active.iloc[validation_end_position:]

    partitions = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    # --------------------------------------------------------
    # Print and collect partition statistics
    # --------------------------------------------------------

    print(f"First attack window : {first_attack_window}")
    print(f"Last attack window  : {last_attack_window}")
    print(f"Active windows      : {number_of_active_windows}")

    results = []

    for partition_name, subset in partitions.items():

        normal_count = int(subset["Normal_Count"].sum())
        attack_count = int(subset["Attack_Count"].sum())

        total_count = normal_count + attack_count

        start_window = int(subset["Window_ID"].min())
        end_window = int(subset["Window_ID"].max())

        start_timestamp = float(
            subset["Start_Timestamp"].min()
        )

        end_timestamp = float(
            subset["End_Timestamp"].max()
        )

        has_normal = normal_count > 0
        has_attack = attack_count > 0

        valid = has_normal and has_attack

        result = {
            "Dataset": dataset_name,
            "Partition": partition_name,
            "Start_Window": start_window,
            "End_Window": end_window,
            "Start_Timestamp": start_timestamp,
            "End_Timestamp": end_timestamp,
            "Normal_Count": normal_count,
            "Attack_Count": attack_count,
            "Total_Count": total_count,
            "Attack_Percentage": (
                attack_count / total_count * 100
                if total_count else 0
            ),
            "Has_Normal": has_normal,
            "Has_Attack": has_attack,
            "Valid_Two_Class_Partition": valid,
        }

        results.append(result)

        print(f"\n{partition_name.upper()}")

        print(
            f"Windows       : {start_window} - {end_window}"
        )

        print(
            f"Normal records: {normal_count:,}"
        )

        print(
            f"Attack records: {attack_count:,}"
        )

        print(
            f"Total records : {total_count:,}"
        )

        print(
            f"Attack ratio  : {result['Attack_Percentage']:.2f}%"
        )

        print(
            f"Class support : {'PASS' if valid else 'FAIL'}"
        )

    # --------------------------------------------------------
    # Identify normal-only follow-up period
    # --------------------------------------------------------

    later_normal = df[
        df["Window_ID"] > last_attack_window
    ].copy()

    later_normal_count = int(
        later_normal["Normal_Count"].sum()
    )

    later_attack_count = int(
        later_normal["Attack_Count"].sum()
    )

    print("\nPOST-ATTACK NORMAL-ONLY PERIOD")

    print(
        f"Windows       : "
        f"{int(later_normal['Window_ID'].min()) if not later_normal.empty else 'N/A'}"
        f" - "
        f"{int(later_normal['Window_ID'].max()) if not later_normal.empty else 'N/A'}"
    )

    print(f"Normal records: {later_normal_count:,}")
    print(f"Attack records: {later_attack_count:,}")

    if later_attack_count == 0 and later_normal_count > 0:
        print("Status        : NORMAL-ONLY FOLLOW-UP")
    else:
        print("Status        : REVIEW REQUIRED")

    return results


# ============================================================
# 4. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 85)
    print("BYTEGUARD-IoV: CHRONOLOGICAL SPLIT VALIDATION")
    print("=" * 85)

    all_results = []

    for dataset_name, filename in FILES.items():

        results = validate_dataset(
            dataset_name=dataset_name,
            filename=filename
        )

        all_results.extend(results)

    # --------------------------------------------------------
    # Save partition plan
    # --------------------------------------------------------

    result_df = pd.DataFrame(all_results)

    csv_path = OUTPUT_DIR / "temporal_partition_plan.csv"

    result_df.to_csv(csv_path, index=False)

    json_path = OUTPUT_DIR / "temporal_partition_plan.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            result_df.to_dict(orient="records"),
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Overall validation
    # --------------------------------------------------------

    print("\n" + "=" * 85)
    print("OVERALL PARTITION VALIDATION")
    print("=" * 85)

    failed = result_df[
        ~result_df["Valid_Two_Class_Partition"]
    ]

    if failed.empty:
        print("RESULT: PASS")
        print("All chronological partitions contain both classes.")
    else:
        print("RESULT: REVIEW REQUIRED")
        print("The following partitions lack one or both classes:")

        print(
            failed[
                [
                    "Dataset",
                    "Partition",
                    "Normal_Count",
                    "Attack_Count"
                ]
            ].to_string(index=False)
        )

    print("\nSaved files:")
    print(csv_path)
    print(json_path)

    print("\n" + "=" * 85)
    print("CHRONOLOGICAL SPLIT VALIDATION COMPLETED")
    print("=" * 85)


if __name__ == "__main__":
    main()
