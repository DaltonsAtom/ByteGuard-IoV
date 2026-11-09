
"""
ByteGuard-IoV — Script 06: Robustness Evaluation

Purpose:
    Evaluate the existing five-class ByteGuard-IoV classifier
    under different input conditions.

Experiments:
    1. Baseline evaluation.
    2. CAN-ID masking sensitivity.
    3. Payload masking sensitivity.
    4. Per-class ROC-AUC.
    5. Precision-recall analysis.
    6. Inference latency.
    7. Model complexity.

Inputs:
    data/processed/test.csv
    models/byteguard_classifier.pt
    models/vocab.json

Outputs:
    outputs/metrics/robustness_results.json
    outputs/metrics/robustness_comparison.csv
    outputs/metrics/roc_auc.json
    outputs/plots/roc_curve.png
    outputs/plots/precision_recall_curve.png
    outputs/plots/robustness_comparison.png

Important:
    Model weights are not modified.
    Masking experiments measure sensitivity, not retrained ablation.
"""

import json
import time

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from sklearn.preprocessing import label_binarize


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

TEST_FILE = DATA_DIR / "test.csv"

MODEL_FILE = MODEL_DIR / "byteguard_classifier.pt"
VOCAB_FILE = MODEL_DIR / "vocab.json"

RESULT_FILE = METRICS_DIR / "robustness_results.json"
COMPARISON_FILE = METRICS_DIR / "robustness_comparison.csv"
ROC_FILE = METRICS_DIR / "roc_auc.json"

ROC_PLOT = PLOTS_DIR / "roc_curve.png"
PR_PLOT = PLOTS_DIR / "precision_recall_curve.png"
COMPARISON_PLOT = PLOTS_DIR / "robustness_comparison.png"

BATCH_SIZE = 128
MAX_LEN = 12

CLASS_NAMES = [
    "Normal",
    "DoS",
    "Fuzzy",
    "Gear Spoofing",
    "RPM Spoofing",
]

NUM_CLASSES = len(CLASS_NAMES)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Create output directories.
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("BYTEGUARD-IoV — ROBUSTNESS EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")


# --------------------------------------------------
# 2. Verify required files
# --------------------------------------------------

required_files = [
    TEST_FILE,
    MODEL_FILE,
    VOCAB_FILE,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# --------------------------------------------------
# 3. Load vocabulary and classifier checkpoint
# --------------------------------------------------

with VOCAB_FILE.open("r", encoding="utf-8") as file:
    vocab = json.load(file)

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=True,
)

# Read architecture parameters from the saved checkpoint.
VOCAB_SIZE = checkpoint["vocab_size"]
EMBED_DIM = checkpoint["embed_dim"]
NUM_HEADS = checkpoint["num_heads"]
NUM_LAYERS = checkpoint["num_layers"]
FEEDFORWARD_DIM = checkpoint["feedforward_dim"]

PAD_ID = vocab["[PAD]"]
UNK_ID = vocab["[UNK]"]
CLS_ID = vocab["[CLS]"]
SEP_ID = vocab["[SEP]"]

# Ensure the saved vocabulary and checkpoint are compatible.
assert len(vocab) == VOCAB_SIZE
assert checkpoint["num_classes"] == NUM_CLASSES

print(f"Vocabulary size: {VOCAB_SIZE:,}")


# --------------------------------------------------
# 4. Load test dataset
# --------------------------------------------------

test_df = pd.read_csv(
    TEST_FILE,
    usecols=["Byte_Sentence", "Class_Label"],
)

test_df["Byte_Sentence"] = (
    test_df["Byte_Sentence"]
    .fillna("")
    .astype(str)
)

test_df["Class_Label"] = (
    test_df["Class_Label"]
    .astype(int)
)

if not test_df["Class_Label"].between(
    0, NUM_CLASSES - 1
).all():
    raise ValueError("Unexpected class labels in test dataset.")

print(f"Test records: {len(test_df):,}")


# --------------------------------------------------
# 5. Tokenization and masking experiments
# --------------------------------------------------

def encode_sentence(sentence, mode="baseline"):
    """
    Convert a CAN byte sentence into token IDs.

    Modes:
        baseline:
            Preserve CAN ID and payload.

        mask_id:
            Replace CAN ID with UNK.

        mask_payload:
            Replace all payload tokens with UNK.

    Special tokens and padding are preserved.
    """

    tokens = sentence.upper().split()

    token_ids = [
        CLS_ID
    ]

    # Token 0 is the CAN identifier.
    can_id = (
        vocab.get(tokens[0], UNK_ID)
        if tokens else UNK_ID
    )

    payload_tokens = tokens[1:]

    if mode == "mask_id":
        can_id = UNK_ID

    token_ids.append(can_id)

    for token in payload_tokens:

        token_id = vocab.get(token, UNK_ID)

        if mode == "mask_payload":
            token_id = UNK_ID

        token_ids.append(token_id)

    token_ids.append(SEP_ID)

    # Enforce the same sequence length for all experiments.
    token_ids = token_ids[:MAX_LEN]

    attention_mask = [1] * len(token_ids)

    padding_length = MAX_LEN - len(token_ids)

    token_ids += [PAD_ID] * padding_length
    attention_mask += [0] * padding_length

    return token_ids, attention_mask


class RobustnessDataset(Dataset):

    def __init__(self, dataframe, mode):

        self.sentences = dataframe[
            "Byte_Sentence"
        ].tolist()

        self.labels = dataframe[
            "Class_Label"
        ].tolist()

        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        token_ids, attention_mask = encode_sentence(
            self.sentences[index],
            mode=self.mode,
        )

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


def create_loader(mode):

    dataset = RobustnessDataset(
        test_df,
        mode,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )


# --------------------------------------------------
# 6. Rebuild the trained classifier architecture
# --------------------------------------------------

class ByteBERTClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM,
            padding_idx=PAD_ID,
        )

        self.position_embedding = nn.Embedding(
            MAX_LEN,
            EMBED_DIM,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS,
        )

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(
            EMBED_DIM,
            NUM_CLASSES,
        )

    def forward(self, input_ids, attention_mask):

        positions = torch.arange(
            input_ids.size(1),
            device=input_ids.device,
        ).unsqueeze(0)

        embeddings = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )

        padding_mask = attention_mask == 0

        hidden_states = self.encoder(
            embeddings,
            src_key_padding_mask=padding_mask,
        )

        cls_representation = hidden_states[:, 0, :]

        return self.classifier(
            self.dropout(cls_representation)
        )


model = ByteBERTClassifier()

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Trained classifier loaded successfully.")


# --------------------------------------------------
# 7. Prediction and evaluation
# --------------------------------------------------

def evaluate_experiment(mode):
    """Evaluate one input condition."""

    loader = create_loader(mode)

    all_predictions = []
    all_probabilities = []
    all_labels = []

    total_time = 0.0
    total_samples = 0

    # Warm-up the model before measuring inference.
    # This prevents initial runtime overhead from dominating.
    warmup_input = torch.full(
        (1, MAX_LEN),
        PAD_ID,
        dtype=torch.long,
        device=DEVICE,
    )

    warmup_mask = torch.ones(
        (1, MAX_LEN),
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        model(warmup_input, warmup_mask)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():

        for input_ids, attention_mask, labels in loader:

            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            logits = model(
                input_ids,
                attention_mask,
            )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            total_time += time.perf_counter() - start_time

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

            total_samples += len(labels)

    all_predictions = np.asarray(all_predictions)
    all_probabilities = np.asarray(all_probabilities)
    all_labels = np.asarray(all_labels)

    results = {
        "experiment": mode,
        "accuracy": accuracy_score(
            all_labels,
            all_predictions,
        ),
        "macro_precision": precision_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        "total_inference_time_seconds": total_time,
        "average_inference_time_ms": (
            total_time / max(total_samples, 1)
        ) * 1000,
        "samples": total_samples,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
        "labels": all_labels,
    }

    return results


# --------------------------------------------------
# 8. Execute three robustness experiments
# --------------------------------------------------

experiment_modes = [
    "baseline",
    "mask_id",
    "mask_payload",
]

experiment_results = {}

for mode in experiment_modes:

    print("\n" + "-" * 60)
    print(f"Running experiment: {mode.upper()}")
    print("-" * 60)

    result = evaluate_experiment(mode)

    experiment_results[mode] = result

    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Macro F1: {result['macro_f1']:.4f}")

    print(
        f"Average inference time: "
        f"{result['average_inference_time_ms']:.4f} ms/frame"
    )


# --------------------------------------------------
# 9. Calculate ROC-AUC and Precision-Recall
# --------------------------------------------------

baseline = experiment_results["baseline"]

true_labels = baseline["labels"]
probabilities = baseline["probabilities"]

binary_labels = label_binarize(
    true_labels,
    classes=list(range(NUM_CLASSES)),
)

roc_results = {}

plt.figure(figsize=(9, 7))

for class_index, class_name in enumerate(CLASS_NAMES):

    false_positive_rate, true_positive_rate, _ = roc_curve(
        binary_labels[:, class_index],
        probabilities[:, class_index],
    )

    class_auc = auc(
        false_positive_rate,
        true_positive_rate,
    )

    roc_results[class_name] = float(class_auc)

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{class_name} (AUC={class_auc:.4f})",
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ByteGuard-IoV — Multiclass ROC Curves")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ROC_PLOT, dpi=300)
plt.close()


# --------------------------------------------------
# 10. Precision-Recall curves
# --------------------------------------------------

pr_results = {}

plt.figure(figsize=(9, 7))

for class_index, class_name in enumerate(CLASS_NAMES):

    precision_values, recall_values, _ = precision_recall_curve(
        binary_labels[:, class_index],
        probabilities[:, class_index],
    )

    average_precision = average_precision_score(
        binary_labels[:, class_index],
        probabilities[:, class_index],
    )

    pr_results[class_name] = float(average_precision)

    plt.plot(
        recall_values,
        precision_values,
        label=f"{class_name} (AP={average_precision:.4f})",
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("ByteGuard-IoV — Precision-Recall Curves")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PR_PLOT, dpi=300)
plt.close()


# --------------------------------------------------
# 11. Model complexity and size
# --------------------------------------------------

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

model_size_bytes = MODEL_FILE.stat().st_size

model_size_mb = model_size_bytes / (1024 ** 2)


# --------------------------------------------------
# 12. Prepare comparative results
# --------------------------------------------------

comparison_rows = []

for mode, result in experiment_results.items():

    comparison_rows.append({
        "Experiment": mode,
        "Accuracy": result["accuracy"],
        "Macro_Precision": result["macro_precision"],
        "Macro_Recall": result["macro_recall"],
        "Macro_F1": result["macro_f1"],
        "Avg_Inference_ms": result["average_inference_time_ms"],
    })

comparison_df = pd.DataFrame(comparison_rows)

comparison_df.to_csv(
    COMPARISON_FILE,
    index=False,
)


# --------------------------------------------------
# 13. Plot robustness comparison
# --------------------------------------------------

plt.figure(figsize=(9, 6))

x_positions = np.arange(len(comparison_df))

plt.bar(
    x_positions - 0.18,
    comparison_df["Accuracy"],
    width=0.36,
    label="Accuracy",
)

plt.bar(
    x_positions + 0.18,
    comparison_df["Macro_F1"],
    width=0.36,
    label="Macro F1",
)

plt.xticks(
    x_positions,
    comparison_df["Experiment"],
)

plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("ByteGuard-IoV — Robustness Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    COMPARISON_PLOT,
    dpi=300,
)

plt.close()


# --------------------------------------------------
# 14. Save detailed JSON report
# --------------------------------------------------

final_report = {
    "test_samples": len(test_df),

    "experiments": {
        mode: {
            key: value
            for key, value in result.items()
            if key not in [
                "predictions",
                "probabilities",
                "labels",
            ]
        }
        for mode, result in experiment_results.items()
    },

    "roc_auc": roc_results,

    "precision_recall_average_precision": pr_results,

    "model_complexity": {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "model_size_bytes": model_size_bytes,
        "model_size_mb": model_size_mb,
    },

    "interpretation": {
        "baseline": "Evaluation using the original CAN ID and payload.",
        "mask_id": (
            "Sensitivity test with CAN ID replaced by UNK. "
            "This is not a separately trained CAN-ID ablation."
        ),
        "mask_payload": (
            "Sensitivity test with payload tokens replaced by UNK. "
            "This is not a separately trained payload ablation."
        ),
    },
}

with RESULT_FILE.open("w", encoding="utf-8") as file:
    json.dump(final_report, file, indent=4)


with ROC_FILE.open("w", encoding="utf-8") as file:
    json.dump(
        {
            "roc_auc": roc_results,
            "average_precision": pr_results,
        },
        file,
        indent=4,
    )


# --------------------------------------------------
# 15. Final summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("BYTEGUARD-IoV — ROBUSTNESS EVALUATION COMPLETE")
print("=" * 70)

print("\nEXPERIMENT COMPARISON")
print(comparison_df.to_string(index=False))

print("\nROC-AUC SCORES")

for class_name, score in roc_results.items():
    print(f"{class_name:20s}: {score:.4f}")

print("\nMODEL COMPLEXITY")
print(f"Total parameters: {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}")
print(f"Checkpoint size: {model_size_mb:.3f} MB")

print("\nGENERATED FILES")
print(f"Results: {RESULT_FILE}")
print(f"Comparison: {COMPARISON_FILE}")
print(f"ROC-AUC: {ROC_FILE}")
print(f"ROC plot: {ROC_PLOT}")
print(f"PR plot: {PR_PLOT}")
print(f"Comparison plot: {COMPARISON_PLOT}")

print("\nAll experiments completed successfully.")
