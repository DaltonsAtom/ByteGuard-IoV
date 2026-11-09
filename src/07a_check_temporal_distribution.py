
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

FILES = {
    "DoS_dataset_processed.csv": "DoS",
    "Fuzzy_dataset_processed.csv": "Fuzzy",
    "gear_dataset_processed.csv": "Gear Spoofing",
    "RPM_dataset_processed.csv": "RPM Spoofing",
}

CHUNK_SIZE = 200000

for filename, attack_name in FILES.items():

    path = DATA_DIR / filename

    print("\n" + "=" * 65)
    print(f"DATASET: {attack_name}")
    print("=" * 65)

    # Find recording timestamp boundaries.
    minimum = float("inf")
    maximum = float("-inf")

    for chunk in pd.read_csv(
        path,
        usecols=["Timestamp"],
        chunksize=CHUNK_SIZE,
    ):
        timestamps = pd.to_numeric(
            chunk["Timestamp"],
            errors="coerce",
        ).dropna()

        if not timestamps.empty:
            minimum = min(minimum, timestamps.min())
            maximum = max(maximum, timestamps.max())

    duration = maximum - minimum

    train_end = minimum + duration * 0.70
    validation_end = minimum + duration * 0.85

    counts = {
        "train": {"Normal": 0, attack_name: 0},
        "validation": {"Normal": 0, attack_name: 0},
        "test": {"Normal": 0, attack_name: 0},
    }

    for chunk in pd.read_csv(
        path,
        usecols=["Timestamp", "Flag"],
        chunksize=CHUNK_SIZE,
    ):

        chunk["Timestamp"] = pd.to_numeric(
            chunk["Timestamp"],
            errors="coerce",
        )

        chunk = chunk.dropna(
            subset=["Timestamp", "Flag"]
        )

        timestamps = chunk["Timestamp"]

        split = pd.Series(
            "test",
            index=chunk.index,
        )

        split.loc[timestamps <= validation_end] = "validation"
        split.loc[timestamps <= train_end] = "train"

        for partition in ["train", "validation", "test"]:

            subset = chunk.loc[split == partition]

            counts[partition]["Normal"] += int(
                (subset["Flag"] == "R").sum()
            )

            counts[partition][attack_name] += int(
                (subset["Flag"] == "T").sum()
            )

    print(f"Timestamp range: {minimum:.6f} - {maximum:.6f}")

    for partition, distribution in counts.items():

        print(f"\n{partition.upper()}")

        for class_name, count in distribution.items():
            print(f"  {class_name:18s}: {count:,}")
