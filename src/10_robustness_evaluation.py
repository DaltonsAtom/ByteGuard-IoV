
"""
BYTEGUARD-IoV
Script 10: Temporal Robustness and Generalisation Audit

Evaluates the trained temporal classifier without retraining.

Experiments:
    1. Baseline evaluation
    2. CAN ID masking
    3. Payload masking
    4. Combined CAN ID + payload masking
    5. Confidence analysis
    6. Per-class performance
    7. CPU inference latency
    8. Model parameter count and checkpoint size

Inputs:
    data/temporal/test.csv
    data/temporal/vocab.json
    models/temporal_byteguard_classifier.pt

Outputs:
    outputs/temporal_metrics/robustness_summary.json
    outputs/temporal_metrics/robustness_comparison.csv
    outputs/temporal_metrics/robustness_per_class.csv
    outputs/temporal_metrics/robustness_predictions.csv
    outputs/temporal_metrics/robustness_latency.json
"""

import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "temporal"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "temporal_metrics"

TEST_PATH = DATA_DIR / "test.csv"
VOCAB_PATH = DATA_DIR / "vocab.json"
CHECKPOINT_PATH = MODEL_DIR / "temporal_byteguard_classifier.pt"

SUMMARY_PATH = OUTPUT_DIR / "robustness_summary.json"
COMPARISON_PATH = OUTPUT_DIR / "robustness_comparison.csv"
PER_CLASS_PATH = OUTPUT_DIR / "robustness_per_class.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "robustness_predictions.csv"
LATENCY_PATH = OUTPUT_DIR / "robustness_latency.json"

BATCH_SIZE = 64
MAX_SEQ_LEN = 12

CLASS_NAMES = [
    "Normal",
    "DoS",
    "Fuzzy",
    "Gear Spoofing",
    "RPM Spoofing"
]

NUM_CLASSES = len(CLASS_NAMES)

RANDOM_SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 3. LOAD VOCABULARY
# ============================================================

def load_vocabulary():

    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        vocab_data = json.load(file)

    token_to_id = vocab_data["token_to_id"]

    required = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    for token in required:
        if token not in token_to_id:
            raise ValueError(f"Missing vocabulary token: {token}")

    return token_to_id


# ============================================================
# 4. LOAD AND TOKENIZE TEST DATA
# ============================================================

def load_test_data(token_to_id):

    df = pd.read_csv(TEST_PATH)

    required_columns = ["Byte_Sentence", "Class_Label"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    pad_id = token_to_id["[PAD]"]
    unk_id = token_to_id["[UNK]"]
    cls_id = token_to_id["[CLS]"]
    sep_id = token_to_id["[SEP]"]

    sequences = []
    original_tokens = []

    for sentence in df["Byte_Sentence"].fillna("").astype(str):

        tokens = sentence.strip().split()

        if len(tokens) < 2:
            raise ValueError(
                "A Byte_Sentence contains fewer than two tokens. "
                "Cannot identify CAN ID and payload positions."
            )

        original_tokens.append(tokens)

        token_ids = [
            token_to_id.get(token, unk_id)
            for token in tokens
        ]

        token_ids = (
            [cls_id]
            + token_ids[:MAX_SEQ_LEN - 2]
            + [sep_id]
        )

        token_ids += [
            pad_id
        ] * (MAX_SEQ_LEN - len(token_ids))

        sequences.append(token_ids)

    labels = pd.to_numeric(
        df["Class_Label"],
        errors="raise"
    ).astype(int).to_numpy()

    if not np.isin(labels, range(NUM_CLASSES)).all():
        raise ValueError("Unexpected class labels detected.")

    return (
        torch.tensor(sequences, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        df,
        original_tokens
    )


# ============================================================
# 5. MODEL ARCHITECTURE
#    Must match Script 09 exactly.
# ============================================================

class ByteGuardBERT(nn.Module):

    def __init__(
        self,
        vocab_size,
        pad_id,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        ffn_dim=128,
        max_seq_len=12,
        dropout=0.1
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_id
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            embed_dim
        )

        self.embedding_norm = nn.LayerNorm(embed_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.mlm_dense = nn.Linear(embed_dim, embed_dim)
        self.mlm_activation = nn.GELU()
        self.mlm_norm = nn.LayerNorm(embed_dim)

        self.mlm_decoder = nn.Linear(
            embed_dim,
            vocab_size,
            bias=False
        )

        self.mlm_decoder.weight = self.token_embedding.weight

        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))

        self.pad_id = pad_id

    def encode(self, input_ids):

        batch_size, sequence_length = input_ids.shape

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)

        embeddings = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )

        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        padding_mask = input_ids.eq(self.pad_id)

        return self.transformer(
            embeddings,
            src_key_padding_mask=padding_mask
        )


class ByteGuardClassifier(nn.Module):

    def __init__(self, encoder, embed_dim=64, num_classes=5):

        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, input_ids):

        hidden_states = self.encoder.encode(input_ids)

        cls_representation = hidden_states[:, 0, :]

        return self.classifier(cls_representation)


# ============================================================
# 6. LOAD TRAINED CLASSIFIER
# ============================================================

def load_model(token_to_id):

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE
        )

    config = checkpoint["model_config"]

    if checkpoint["vocabulary"] != token_to_id:
        raise ValueError(
            "Checkpoint vocabulary does not match vocab.json."
        )

    encoder = ByteGuardBERT(
        vocab_size=config["vocab_size"],
        pad_id=config["pad_id"],
        embed_dim=config["embed_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ffn_dim=config["ffn_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"]
    )

    model = ByteGuardClassifier(
        encoder,
        embed_dim=config["embed_dim"],
        num_classes=NUM_CLASSES
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    return model, config


# ============================================================
# 7. FEATURE MASKING
# ============================================================

def create_masked_inputs(
    input_tensor,
    pad_id,
    mask_id,
    experiment
):
    """
    Sequence layout:

        Position 0  : [CLS]
        Position 1  : CAN ID
        Positions 2+: Payload tokens
        Final token : [SEP] or [PAD]

    The script assumes the first content token is CAN ID.
    """

    masked = input_tensor.clone()

    for row in range(masked.size(0)):

        sequence = masked[row]

        valid_positions = [
            position
            for position in range(1, sequence.size(0))
            if sequence[position].item() != pad_id
            and sequence[position].item() != token_to_id_global["[SEP]"]
        ]

        if not valid_positions:
            raise ValueError("No content tokens found in sequence.")

        can_id_position = valid_positions[0]
        payload_positions = valid_positions[1:]

        if experiment in ("can_id", "combined"):

            masked[row, can_id_position] = mask_id

        if experiment in ("payload", "combined"):

            for position in payload_positions:
                masked[row, position] = mask_id

    return masked


# ============================================================
# 8. MODEL PREDICTIONS
# ============================================================

def predict(model, input_tensor):

    loader = DataLoader(
        TensorDataset(input_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    predictions = []
    probabilities = []

    model.eval()

    with torch.no_grad():

        for (batch,) in loader:

            logits = model(batch.to(DEVICE))

            probs = torch.softmax(logits, dim=1)

            predictions.extend(
                torch.argmax(probs, dim=1).cpu().numpy()
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

    return np.array(predictions), np.array(probabilities)


# ============================================================
# 9. METRIC CALCULATION
# ============================================================

def calculate_metrics(y_true, y_pred, probabilities, experiment):

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    confidence = probabilities.max(axis=1)

    return {
        "experiment": experiment,
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "mean_confidence": float(confidence.mean()),
        "minimum_confidence": float(confidence.min()),
        "maximum_confidence": float(confidence.max()),
        "correct_predictions": int(np.sum(y_true == y_pred)),
        "incorrect_predictions": int(np.sum(y_true != y_pred))
    }


# ============================================================
# 10. PER-CLASS METRICS
# ============================================================

def calculate_per_class_metrics(y_true, predictions_by_experiment):

    rows = []

    for experiment, predictions in predictions_by_experiment.items():

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            predictions,
            labels=list(range(NUM_CLASSES)),
            zero_division=0
        )

        for class_id, class_name in enumerate(CLASS_NAMES):

            rows.append({
                "experiment": experiment,
                "class": class_name,
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1_score": float(f1[class_id]),
                "support": int(support[class_id])
            })

    return pd.DataFrame(rows)


# ============================================================
# 11. INFERENCE LATENCY
# ============================================================

def measure_latency(model, input_tensor):

    sample_size = min(1000, len(input_tensor))

    sample = input_tensor[:sample_size]

    # Warm-up.
    with torch.no_grad():
        for _ in range(5):
            model(sample[:min(BATCH_SIZE, sample_size)])

    # Single-record latency.
    single_sample = sample[:1]

    single_times = []

    with torch.no_grad():

        for _ in range(100):

            start = time.perf_counter()

            model(single_sample)

            elapsed = time.perf_counter() - start

            single_times.append(elapsed)

    # Batched throughput.
    batch_loader = DataLoader(
        TensorDataset(sample),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    total_records = 0

    start = time.perf_counter()

    with torch.no_grad():

        for (batch,) in batch_loader:

            model(batch)

            total_records += len(batch)

    total_batch_time = time.perf_counter() - start

    average_single_ms = (
        np.mean(single_times) * 1000
    )

    median_single_ms = (
        np.median(single_times) * 1000
    )

    batch_ms_per_record = (
        total_batch_time / max(total_records, 1)
    ) * 1000

    throughput = (
        total_records / max(total_batch_time, 1e-12)
    )

    return {
        "device": str(DEVICE),
        "single_record_mean_latency_ms": float(average_single_ms),
        "single_record_median_latency_ms": float(median_single_ms),
        "batched_latency_per_record_ms": float(batch_ms_per_record),
        "batched_throughput_records_per_second": float(throughput),
        "latency_scope": (
            "Neural network forward pass only. "
            "Excludes CSV loading, tokenization, CAN acquisition "
            "and communication overhead."
        )
    }


# ============================================================
# 12. MAIN
# ============================================================

def main():

    global token_to_id_global

    set_seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("BYTEGUARD-IoV: TEMPORAL ROBUSTNESS EVALUATION")
    print("=" * 75)

    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Load vocabulary and dataset
    # --------------------------------------------------------

    print("\nLoading vocabulary...")

    token_to_id = load_vocabulary()

    token_to_id_global = token_to_id

    pad_id = token_to_id["[PAD]"]
    mask_id = token_to_id["[MASK]"]

    print(f"Vocabulary size: {len(token_to_id)}")

    print("\nLoading temporal test dataset...")

    input_tensor, labels, test_df, original_tokens = load_test_data(
        token_to_id
    )

    y_true = labels.numpy()

    print(f"Test records: {len(input_tensor):,}")

    # --------------------------------------------------------
    # Validate token structure
    # --------------------------------------------------------

    print("\nChecking sentence structure...")

    print("Example Byte_Sentence:")
    print(test_df["Byte_Sentence"].iloc[0])

    print("Example token count:", len(original_tokens[0]))

    print(
        "Assumption: first content token = CAN ID; "
        "remaining content tokens = payload."
    )

    # --------------------------------------------------------
    # Load classifier
    # --------------------------------------------------------

    print("\nLoading temporal classifier...")

    model, config = load_model(token_to_id)

    print("Checkpoint loaded successfully.")

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    checkpoint_size_mb = (
        CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
    )

    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Checkpoint size: {checkpoint_size_mb:.4f} MB")

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("EXPERIMENT 1: BASELINE")
    print("=" * 75)

    predictions_by_experiment = {}
    probabilities_by_experiment = {}
    summary_rows = []

    baseline_predictions, baseline_probabilities = predict(
        model,
        input_tensor
    )

    predictions_by_experiment["baseline"] = baseline_predictions
    probabilities_by_experiment["baseline"] = baseline_probabilities

    baseline_metrics = calculate_metrics(
        y_true,
        baseline_predictions,
        baseline_probabilities,
        "baseline"
    )

    summary_rows.append(baseline_metrics)

    print(f"Accuracy : {baseline_metrics['accuracy']:.6f}")
    print(f"Macro F1 : {baseline_metrics['macro_f1']:.6f}")
    print(f"Confidence: {baseline_metrics['mean_confidence']:.6f}")

    # --------------------------------------------------------
    # Feature masking experiments
    # --------------------------------------------------------

    experiments = [
        ("can_id", "CAN ID masking"),
        ("payload", "Payload masking"),
        ("combined", "Combined CAN ID + payload masking")
    ]

    for experiment, description in experiments:

        print("\n" + "=" * 75)
        print(f"EXPERIMENT: {description.upper()}")
        print("=" * 75)

        masked_inputs = create_masked_inputs(
            input_tensor,
            pad_id,
            mask_id,
            experiment
        )

        predictions, probabilities = predict(
            model,
            masked_inputs
        )

        predictions_by_experiment[experiment] = predictions
        probabilities_by_experiment[experiment] = probabilities

        metrics = calculate_metrics(
            y_true,
            predictions,
            probabilities,
            experiment
        )

        summary_rows.append(metrics)

        print(f"Accuracy : {metrics['accuracy']:.6f}")
        print(f"Macro F1 : {metrics['macro_f1']:.6f}")
        print(f"Confidence: {metrics['mean_confidence']:.6f}")

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(summary_rows)

    baseline_accuracy = baseline_metrics["accuracy"]
    baseline_f1 = baseline_metrics["macro_f1"]

    comparison_df["accuracy_drop"] = (
        baseline_accuracy - comparison_df["accuracy"]
    )

    comparison_df["macro_f1_drop"] = (
        baseline_f1 - comparison_df["macro_f1"]
    )

    comparison_df.to_csv(
        COMPARISON_PATH,
        index=False
    )

    print("\n" + "=" * 75)
    print("ROBUSTNESS COMPARISON")
    print("=" * 75)

    print(
        comparison_df[
            [
                "experiment",
                "accuracy",
                "macro_f1",
                "mean_confidence",
                "accuracy_drop",
                "macro_f1_drop"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Per-class analysis
    # --------------------------------------------------------

    per_class_df = calculate_per_class_metrics(
        y_true,
        predictions_by_experiment
    )

    per_class_df.to_csv(
        PER_CLASS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Prediction-level error analysis
    # --------------------------------------------------------

    prediction_output = test_df.copy()

    prediction_output["True_Label"] = y_true

    for experiment, predictions in predictions_by_experiment.items():

        prediction_output[
            f"Prediction_{experiment}"
        ] = predictions

        prediction_output[
            f"Confidence_{experiment}"
        ] = probabilities_by_experiment[experiment].max(axis=1)

        prediction_output[
            f"Correct_{experiment}"
        ] = predictions == y_true

    prediction_output["True_Class"] = [
        CLASS_NAMES[value] for value in y_true
    ]

    prediction_output.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Latency evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("INFERENCE LATENCY EVALUATION")
    print("=" * 75)

    latency_results = measure_latency(
        model,
        input_tensor
    )

    for key, value in latency_results.items():
        print(f"{key}: {value}")

    latency_results["trainable_parameters"] = parameter_count
    latency_results["checkpoint_size_mb"] = checkpoint_size_mb

    with open(LATENCY_PATH, "w", encoding="utf-8") as file:
        json.dump(
            latency_results,
            file,
            indent=4,
            allow_nan=False
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = {
        "experiment": "ByteGuard-IoV Temporal Robustness Evaluation",
        "test_records": len(y_true),
        "class_names": CLASS_NAMES,
        "feature_masking_assumption": (
            "First content token is CAN ID; subsequent content "
            "tokens are payload tokens. This must be verified "
            "against the actual Byte_Sentence construction."
        ),
        "baseline": baseline_metrics,
        "experiments": summary_rows,
        "latency": latency_results,
        "interpretation": {
            "masking": (
                "Feature masking measures sensitivity of the trained "
                "model to altered inputs. It is not equivalent to "
                "training a separate model without those features."
            ),
            "generalisation": (
                "This experiment evaluates robustness on the existing "
                "temporal test set. It does not establish generalisation "
                "to independent vehicle or recording sessions."
            )
        }
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            allow_nan=False
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("SCRIPT 10 COMPLETED")
    print("=" * 75)

    print(f"Baseline accuracy: {baseline_accuracy:.6f}")
    print(f"Baseline macro F1: {baseline_f1:.6f}")

    print("\nGenerated files:")

    print(SUMMARY_PATH)
    print(COMPARISON_PATH)
    print(PER_CLASS_PATH)
    print(PREDICTIONS_PATH)
    print(LATENCY_PATH)


if __name__ == "__main__":
    main()
