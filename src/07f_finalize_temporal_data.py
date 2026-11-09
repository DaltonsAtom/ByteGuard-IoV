
import pandas as pd
import json
import numpy as np
from pathlib import Path
from collections import Counter

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "temporal"
OUTPUT_DIR = ROOT / "outputs" / "temporal_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": DATA_DIR / "train.csv",
    "validation": DATA_DIR / "validation.csv",
    "test": DATA_DIR / "test.csv",
    "normal_followup": DATA_DIR / "normal_followup.csv",
}

EXPECTED_CLASSES = {
    "Normal": 0,
    "DoS": 1,
    "Fuzzy": 2,
    "Gear Spoofing": 3,
    "RPM Spoofing": 4,
}

EXPECTED_SIZES = {
    "train": 10_000,
    "validation": 1_500,
    "test": 1_500,
}

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
]

REQUIRED_COLUMNS = [
    "Timestamp",
    "Flag",
    "Class_Label",
    "Byte_Sentence",
    "Source_Dataset",
    "Class_Name",
]


# ============================================================
# 2. JSON SAFE CONVERSION
# ============================================================

def make_json_serializable(obj):

    """
    Convert NumPy and Pandas values into native Python
    types so that json.dump() can serialize them.
    """

    if isinstance(obj, dict):

        return {
            str(key): make_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):

        return [
            make_json_serializable(value)
            for value in obj
        ]

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()

    return obj


# ============================================================
# 3. LOAD GENERATED DATASETS
# ============================================================

def load_datasets():

    datasets = {}

    for name, path in FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {path}"
            )

        df = pd.read_csv(path)

        missing = set(REQUIRED_COLUMNS) - set(df.columns)

        if missing:
            raise ValueError(
                f"{name}: Missing columns {sorted(missing)}"
            )

        df["Timestamp"] = pd.to_numeric(
            df["Timestamp"],
            errors="coerce"
        )

        df["Class_Label"] = pd.to_numeric(
            df["Class_Label"],
            errors="coerce"
        )

        if df[
            ["Timestamp", "Class_Label", "Byte_Sentence"]
        ].isna().any().any():

            raise ValueError(
                f"{name}: Missing values detected in required fields."
            )

        df["Class_Label"] = df["Class_Label"].astype(int)

        datasets[name] = df

        print(
            f"Loaded {name:16s}: {len(df):,} records"
        )

    return datasets


# ============================================================
# 4. VALIDATE LABELS AND CLASS DISTRIBUTION
# ============================================================

def validate_labels(datasets):

    print("\n" + "=" * 75)
    print("LABEL AND CLASS VALIDATION")
    print("=" * 75)

    report = {}

    for name, df in datasets.items():

        print(f"\n{name.upper()}")

        if name == "normal_followup":

            valid = (
                (df["Class_Label"] == 0) &
                (df["Class_Name"] == "Normal") &
                (df["Flag"] == "R")
            )

            if not valid.all():
                raise ValueError(
                    "Normal follow-up dataset contains non-normal records."
                )

            print("PASS: All records are normal.")

            report[name] = {
                "total_records": int(len(df)),
                "class_distribution": (
                    df["Class_Name"].value_counts().to_dict()
                ),
                "label_validation": "PASS",
            }

            continue

        expected_count = EXPECTED_SIZES[name]

        if len(df) != expected_count * 5:
            raise ValueError(
                f"{name}: Expected {expected_count * 5:,} records, "
                f"found {len(df):,}."
            )

        for class_name, class_id in EXPECTED_CLASSES.items():

            subset = df[df["Class_Name"] == class_name]

            if len(subset) != expected_count:
                raise ValueError(
                    f"{name}: Incorrect count for {class_name}."
                )

            if not (subset["Class_Label"] == class_id).all():
                raise ValueError(
                    f"{name}: Incorrect numeric labels for {class_name}."
                )

            expected_flag = "R" if class_id == 0 else "T"

            if not (subset["Flag"] == expected_flag).all():
                raise ValueError(
                    f"{name}: Incorrect Flag values for {class_name}."
                )

            print(
                f"{class_name:18s}: {len(subset):,} PASS"
            )

        report[name] = {
            "total_records": int(len(df)),
            "class_distribution": (
                df["Class_Name"].value_counts().to_dict()
            ),
            "label_validation": "PASS",
        }

    return report


# ============================================================
# 5. CHECK CHRONOLOGICAL SEPARATION
# ============================================================

def validate_chronology(datasets):

    print("\n" + "=" * 75)
    print("CHRONOLOGICAL SEPARATION VALIDATION")
    print("=" * 75)

    temporal_report = {}

    sources = sorted(
        datasets["train"]["Source_Dataset"].unique()
    )

    for source in sources:

        train = datasets["train"]
        validation = datasets["validation"]
        test = datasets["test"]
        followup = datasets["normal_followup"]

        source_train = train[
            train["Source_Dataset"] == source
        ]

        source_validation = validation[
            validation["Source_Dataset"] == source
        ]

        source_test = test[
            test["Source_Dataset"] == source
        ]

        source_followup = followup[
            followup["Source_Dataset"] == source
        ]

        if any(
            frame.empty
            for frame in [
                source_train,
                source_validation,
                source_test,
                source_followup,
            ]
        ):
            raise ValueError(
                f"{source}: Missing records in one or more partitions."
            )

        train_max = float(
            source_train["Timestamp"].max()
        )

        validation_min = float(
            source_validation["Timestamp"].min()
        )

        validation_max = float(
            source_validation["Timestamp"].max()
        )

        test_min = float(
            source_test["Timestamp"].min()
        )

        test_max = float(
            source_test["Timestamp"].max()
        )

        followup_min = float(
            source_followup["Timestamp"].min()
        )

        checks = {
            "train_before_validation": bool(
                train_max < validation_min
            ),
            "validation_before_test": bool(
                validation_max < test_min
            ),
            "test_before_followup": bool(
                test_max < followup_min
            ),
        }

        if not all(checks.values()):
            raise ValueError(
                f"{source}: Chronological overlap detected: {checks}"
            )

        temporal_report[source] = {
            "train_max_timestamp": train_max,
            "validation_min_timestamp": validation_min,
            "validation_max_timestamp": validation_max,
            "test_min_timestamp": test_min,
            "test_max_timestamp": test_max,
            "followup_min_timestamp": followup_min,
            "checks": checks,
        }

        print(f"\n{source}")

        for check_name, passed in checks.items():

            print(
                f"  {check_name:30s}: "
                f"{'PASS' if passed else 'FAIL'}"
            )

    return temporal_report


# ============================================================
# 6. CHECK CROSS-PARTITION DUPLICATES
# ============================================================

def validate_duplicates(datasets):

    print("\n" + "=" * 75)
    print("CROSS-PARTITION DUPLICATE VALIDATION")
    print("=" * 75)

    partitions = [
        "train",
        "validation",
        "test",
    ]

    key_sets = {}

    for name in partitions:

        df = datasets[name]

        keys = set(
            zip(
                df["Source_Dataset"],
                df["Timestamp"],
                df["Byte_Sentence"],
            )
        )

        key_sets[name] = keys

    duplicate_report = {}

    for i in range(len(partitions)):

        for j in range(i + 1, len(partitions)):

            first = partitions[i]
            second = partitions[j]

            shared = len(
                key_sets[first] & key_sets[second]
            )

            duplicate_report[
                f"{first}_vs_{second}"
            ] = int(shared)

            print(
                f"{first:12s} vs {second:12s}: "
                f"{shared:,} shared records"
            )

            if shared > 0:
                raise ValueError(
                    f"Exact duplicate records found between "
                    f"{first} and {second}."
                )

    return duplicate_report


# ============================================================
# 7. BUILD TRAINING-ONLY VOCABULARY
# ============================================================

def build_vocabulary(train_df):

    print("\n" + "=" * 75)
    print("BUILDING TRAINING-ONLY VOCABULARY")
    print("=" * 75)

    token_counts = Counter()

    for sentence in train_df["Byte_Sentence"]:

        tokens = str(sentence).split()

        token_counts.update(tokens)

    token_to_id = {
        token: index
        for index, token in enumerate(SPECIAL_TOKENS)
    }

    observed_tokens = sorted(
        set(token_counts.keys()) - set(SPECIAL_TOKENS)
    )

    for token in observed_tokens:

        token_to_id[token] = len(token_to_id)

    id_to_token = [
        token
        for token, index in sorted(
            token_to_id.items(),
            key=lambda item: item[1]
        )
    ]

    vocabulary = {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "special_tokens": {
            "pad": "[PAD]",
            "unk": "[UNK]",
            "cls": "[CLS]",
            "sep": "[SEP]",
            "mask": "[MASK]",
        },
        "vocab_size": len(token_to_id),
        "tokenization": "whitespace",
        "built_from": "data/temporal/train.csv",
        "unique_observed_tokens": len(observed_tokens),
        "training_token_count": int(
            sum(token_counts.values())
        ),
    }

    vocab_path = DATA_DIR / "vocab.json"

    with open(
        vocab_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            vocabulary,
            file,
            indent=4
        )

    print(
        f"Unique observed tokens: {len(observed_tokens):,}"
    )

    print(
        f"Special tokens        : {len(SPECIAL_TOKENS)}"
    )

    print(
        f"Total vocabulary size : {len(token_to_id):,}"
    )

    print(f"\nVocabulary saved: {vocab_path}")

    return vocabulary


# ============================================================
# 8. CALCULATE OOV RATES
# ============================================================

def calculate_oov_rates(datasets, vocabulary):

    print("\n" + "=" * 75)
    print("OUT-OF-VOCABULARY TOKEN ANALYSIS")
    print("=" * 75)

    known_tokens = set(
        vocabulary["token_to_id"].keys()
    )

    report = {}

    for name in [
        "validation",
        "test",
        "normal_followup",
    ]:

        total_tokens = 0
        unknown_tokens = 0

        for sentence in datasets[name]["Byte_Sentence"]:

            tokens = str(sentence).split()

            total_tokens += len(tokens)

            unknown_tokens += sum(
                token not in known_tokens
                for token in tokens
            )

        oov_rate = (
            unknown_tokens / total_tokens * 100
            if total_tokens else 0
        )

        report[name] = {
            "total_tokens": int(total_tokens),
            "unknown_tokens": int(unknown_tokens),
            "oov_rate_percent": float(oov_rate),
        }

        print(f"\n{name.upper()}")

        print(f"Total tokens   : {total_tokens:,}")
        print(f"Unknown tokens : {unknown_tokens:,}")
        print(f"OOV rate       : {oov_rate:.4f}%")

    return report


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 80)
    print("BYTEGUARD-IoV: FINAL TEMPORAL DATA VALIDATION")
    print("=" * 80)

    datasets = load_datasets()

    label_report = validate_labels(datasets)

    temporal_report = validate_chronology(datasets)

    duplicate_report = validate_duplicates(datasets)

    vocabulary = build_vocabulary(
        datasets["train"]
    )

    oov_report = calculate_oov_rates(
        datasets,
        vocabulary
    )

    final_report = {
        "project": "ByteGuard-IoV",
        "evaluation_type": "Chronological temporal evaluation",
        "label_validation": label_report,
        "chronological_validation": temporal_report,
        "duplicate_validation": duplicate_report,
        "vocabulary": {
            "vocab_size": int(vocabulary["vocab_size"]),
            "built_from": vocabulary["built_from"],
            "tokenization": vocabulary["tokenization"],
        },
        "oov_analysis": oov_report,
        "overall_status": "PASS",
    }

    # Convert all remaining NumPy/Pandas values before saving.
    final_report = make_json_serializable(final_report)

    report_path = (
        DATA_DIR / "temporal_final_audit.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            final_report,
            file,
            indent=4,
            allow_nan=False
        )

    print("\n" + "=" * 80)
    print("FINAL TEMPORAL DATA VALIDATION")
    print("=" * 80)

    print("Class distribution        : PASS")
    print("Numeric labels            : PASS")
    print("Chronological separation  : PASS")
    print("Cross-partition duplicates: PASS")
    print("Normal follow-up dataset  : PASS")
    print("Training-only vocabulary  : PASS")

    print("\nGenerated files:")
    print(DATA_DIR / "vocab.json")
    print(report_path)

    print("\n" + "=" * 80)
    print("SCRIPT 07 COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
