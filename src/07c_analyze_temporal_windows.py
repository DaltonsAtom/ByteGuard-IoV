
import pandas as pd
from pathlib import Path
import numpy as np

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs" / "temporal_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 200_000
WINDOW_SECONDS = 5

FILES = {
    "DoS_dataset_processed.csv": "DoS",
    "Fuzzy_dataset_processed.csv": "Fuzzy",
    "gear_dataset_processed.csv": "Gear Spoofing",
    "RPM_dataset_processed.csv": "RPM Spoofing",
}


# ============================================================
# 2. ANALYSE ONE DATASET
# ============================================================

def analyze_dataset(filename, attack_name):

    path = DATA_DIR / filename

    print("\n" + "=" * 80)
    print(f"DATASET: {attack_name}")
    print("=" * 80)

    if not path.exists():
        print(f"ERROR: Dataset not found: {path}")
        return

    # --------------------------------------------------------
    # Pass 1: Find timestamp range
    # --------------------------------------------------------

    minimum = float("inf")
    maximum = float("-inf")

    for chunk in pd.read_csv(
        path,
        usecols=["Timestamp"],
        chunksize=CHUNK_SIZE
    ):

        timestamps = pd.to_numeric(
            chunk["Timestamp"],
            errors="coerce"
        ).dropna()

        if not timestamps.empty:

            minimum = min(minimum, timestamps.min())
            maximum = max(maximum, timestamps.max())

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        print("ERROR: No valid timestamps found.")
        return

    print(f"Timestamp start: {minimum:.6f}")
    print(f"Timestamp end  : {maximum:.6f}")

    # --------------------------------------------------------
    # Create fixed-width temporal windows
    # --------------------------------------------------------

    number_of_windows = (
        int(np.ceil((maximum - minimum) / WINDOW_SECONDS)) + 1
    )

    # Each window stores normal and attack counts
    normal_counts = np.zeros(number_of_windows, dtype=np.int64)
    attack_counts = np.zeros(number_of_windows, dtype=np.int64)

    # --------------------------------------------------------
    # Pass 2: Count records in each temporal window
    # --------------------------------------------------------

    for chunk in pd.read_csv(
        path,
        usecols=["Timestamp", "Flag"],
        chunksize=CHUNK_SIZE
    ):

        chunk["Timestamp"] = pd.to_numeric(
            chunk["Timestamp"],
            errors="coerce"
        )

        chunk = chunk.dropna(
            subset=["Timestamp", "Flag"]
        )

        # Calculate window index for every record
        window_indices = (
            (chunk["Timestamp"] - minimum) // WINDOW_SECONDS
        ).astype(int)

        # Protect against floating-point boundary issues
        window_indices = window_indices.clip(
            lower=0,
            upper=number_of_windows - 1
        )

        # Count normal records
        normal_mask = chunk["Flag"] == "R"

        normal_window_counts = (
            window_indices[normal_mask]
            .value_counts()
        )

        for index, count in normal_window_counts.items():
            normal_counts[int(index)] += int(count)

        # Count attack records
        attack_mask = chunk["Flag"] == "T"

        attack_window_counts = (
            window_indices[attack_mask]
            .value_counts()
        )

        for index, count in attack_window_counts.items():
            attack_counts[int(index)] += int(count)

    # --------------------------------------------------------
    # Pass 3: Construct window-level report
    # --------------------------------------------------------

    rows = []

    for index in range(number_of_windows):

        window_start = minimum + index * WINDOW_SECONDS
        window_end = window_start + WINDOW_SECONDS

        normal = int(normal_counts[index])
        attack = int(attack_counts[index])

        total = normal + attack

        if normal > 0 and attack > 0:
            window_type = "Mixed"

        elif attack > 0:
            window_type = "Attack Only"

        elif normal > 0:
            window_type = "Normal Only"

        else:
            window_type = "Empty"

        rows.append({
            "Window_ID": index,
            "Start_Timestamp": window_start,
            "End_Timestamp": min(window_end, maximum),
            "Normal_Count": normal,
            "Attack_Count": attack,
            "Total_Count": total,
            "Attack_Percentage": (
                (attack / total) * 100 if total > 0 else 0
            ),
            "Window_Type": window_type
        })

    report = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Save full window report
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        f"{attack_name.lower().replace(' ', '_')}_5sec_windows.csv"
    )

    report.to_csv(output_path, index=False)

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    active = report[report["Total_Count"] > 0]

    mixed = report[report["Window_Type"] == "Mixed"]

    normal_only = report[
        report["Window_Type"] == "Normal Only"
    ]

    attack_only = report[
        report["Window_Type"] == "Attack Only"
    ]

    print("\nTEMPORAL WINDOW SUMMARY")

    print(f"Window duration       : {WINDOW_SECONDS} seconds")
    print(f"Total windows         : {len(report):,}")
    print(f"Non-empty windows     : {len(active):,}")
    print(f"Mixed windows         : {len(mixed):,}")
    print(f"Normal-only windows   : {len(normal_only):,}")
    print(f"Attack-only windows   : {len(attack_only):,}")

    print("\nRECORD COUNTS")

    print(f"Normal records        : {normal_counts.sum():,}")
    print(f"Attack records        : {attack_counts.sum():,}")

    print("\nATTACK DISTRIBUTION")

    if not mixed.empty or not attack_only.empty:

        attack_windows = report[
            report["Attack_Count"] > 0
        ]

        print(
            "First attack window  : "
            f"{attack_windows.iloc[0]['Start_Timestamp']:.6f}"
        )

        print(
            "Last attack window   : "
            f"{attack_windows.iloc[-1]['End_Timestamp']:.6f}"
        )

    else:
        print("No attack windows found.")

    # Display windows with the highest attack counts
    print("\nTOP 10 WINDOWS BY ATTACK COUNT")

    top_windows = report.nlargest(
        10,
        "Attack_Count"
    )

    print(
        top_windows[
            [
                "Window_ID",
                "Start_Timestamp",
                "End_Timestamp",
                "Normal_Count",
                "Attack_Count",
                "Attack_Percentage",
                "Window_Type"
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved report: {output_path}")


# ============================================================
# 3. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 80)
    print("BYTEGUARD-IoV: 5-SECOND TEMPORAL TRAFFIC ANALYSIS")
    print("=" * 80)

    print(f"\nWindow duration: {WINDOW_SECONDS} seconds")
    print(f"Chunk size     : {CHUNK_SIZE:,}")

    for filename, attack_name in FILES.items():

        analyze_dataset(
            filename=filename,
            attack_name=attack_name
        )

    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS COMPLETED")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
