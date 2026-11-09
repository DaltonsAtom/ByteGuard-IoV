
"""
BYTEGUARD-IoV
Script 09: Temporal Supervised Fine-Tuning

Inputs:
    data/temporal/train.csv
    data/temporal/validation.csv
    data/temporal/test.csv
    data/temporal/normal_followup.csv
    data/temporal/vocab.json
    models/temporal_byteguard_mlm.pt

Outputs:
    models/temporal_byteguard_classifier.pt
    outputs/temporal_metrics/temporal_classification_history.json
    outputs/temporal_metrics/temporal_test_classification_report.json
    outputs/temporal_metrics/temporal_test_confusion_matrix.csv
    outputs/temporal_metrics/temporal_followup_report.json
    outputs/temporal_metrics/temporal_test_predictions.csv
    outputs/temporal_metrics/temporal_followup_predictions.csv
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "temporal"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "temporal_metrics"

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"
FOLLOWUP_PATH = DATA_DIR / "normal_followup.csv"
VOCAB_PATH = DATA_DIR / "vocab.json"

PRETRAINED_PATH = MODEL_DIR / "temporal_byteguard_mlm.pt"
CLASSIFIER_PATH = MODEL_DIR / "temporal_byteguard_classifier.pt"

HISTORY_PATH = OUTPUT_DIR / "temporal_classification_history.json"
REPORT_PATH = OUTPUT_DIR / "temporal_test_classification_report.json"
CONFUSION_PATH = OUTPUT_DIR / "temporal_test_confusion_matrix.csv"
FOLLOWUP_REPORT_PATH = OUTPUT_DIR / "temporal_followup_report.json"

TEST_PREDICTIONS_PATH = OUTPUT_DIR / "temporal_test_predictions.csv"
FOLLOWUP_PREDICTIONS_PATH = OUTPUT_DIR / "temporal_followup_predictions.csv"

# Model architecture: must match Script 08.
EMBED_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 4
FFN_DIM = 128
MAX_SEQ_LEN = 12
DROPOUT = 0.1

# Fine-tuning configuration.
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
PATIENCE = 3

RANDOM_SEED = 42

CLASS_NAMES = [
    "Normal",
    "DoS",
    "Fuzzy",
    "Gear Spoofing",
    "RPM Spoofing"
]

NUM_CLASSES = len(CLASS_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. LOAD VOCABULARY
# ============================================================

def load_vocabulary():

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Vocabulary not found: {VOCAB_PATH}")

    with open(VOCAB_PATH, "r", encoding="utf-8") as file:
        vocab_data = json.load(file)

    token_to_id = vocab_data["token_to_id"]

    required_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    for token in required_tokens:
        if token not in token_to_id:
            raise ValueError(f"Missing special token: {token}")

    vocab_size = len(token_to_id)

    if vocab_data.get("vocab_size", vocab_size) != vocab_size:
        raise ValueError("Vocabulary size mismatch.")

    print(f"Vocabulary size: {vocab_size}")

    return token_to_id


# ============================================================
# 4. DATASET ENCODING
# ============================================================

def load_dataset(csv_path, token_to_id, require_labels=True):

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Byte_Sentence" not in df.columns:
        raise ValueError(f"Byte_Sentence missing in {csv_path.name}")

    if require_labels and "Class_Label" not in df.columns:
        raise ValueError(f"Class_Label missing in {csv_path.name}")

    pad_id = token_to_id["[PAD]"]
    unk_id = token_to_id["[UNK]"]
    cls_id = token_to_id["[CLS]"]
    sep_id = token_to_id["[SEP]"]

    encoded_sequences = []

    for sentence in df["Byte_Sentence"].fillna("").astype(str):

        tokens = sentence.strip().split()

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

        encoded_sequences.append(token_ids)

    input_tensor = torch.tensor(
        encoded_sequences,
        dtype=torch.long
    )

    if require_labels:

        labels = pd.to_numeric(
            df["Class_Label"],
            errors="raise"
        ).astype(int)

        if not labels.isin(range(NUM_CLASSES)).all():
            raise ValueError(
                f"Unexpected class labels in {csv_path.name}: "
                f"{sorted(labels.unique().tolist())}"
            )

        label_tensor = torch.tensor(
            labels.to_numpy(),
            dtype=torch.long
        )

        print(f"{csv_path.name:<25}: {len(df):,} records")
        print(f"  Class distribution: {labels.value_counts().sort_index().to_dict()}")

        return input_tensor, label_tensor, df

    print(f"{csv_path.name:<25}: {len(df):,} records")

    return input_tensor, df


# ============================================================
# 5. BYTEGUARD TRANSFORMER
#    Architecture and parameter names match Script 08.
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
        self.max_seq_len = max_seq_len

    def encode(self, input_ids):

        batch_size, sequence_length = input_ids.shape

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)

        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)

        embeddings = token_embeddings + position_embeddings

        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        padding_mask = input_ids.eq(self.pad_id)

        hidden_states = self.transformer(
            embeddings,
            src_key_padding_mask=padding_mask
        )

        return hidden_states

    def forward(self, input_ids):

        hidden_states = self.encode(input_ids)

        mlm_output = self.mlm_dense(hidden_states)
        mlm_output = self.mlm_activation(mlm_output)
        mlm_output = self.mlm_norm(mlm_output)

        logits = self.mlm_decoder(mlm_output) + self.mlm_bias

        return logits


# ============================================================
# 6. CLASSIFICATION MODEL
# ============================================================

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

        # Use the [CLS] representation.
        cls_representation = hidden_states[:, 0, :]

        logits = self.classifier(cls_representation)

        return logits


# ============================================================
# 7. TRAINING / VALIDATION
# ============================================================

def run_epoch(
    model,
    dataloader,
    loss_function,
    optimizer=None
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_records = 0

    for input_ids, labels in dataloader:

        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            logits = model(input_ids)

            loss = loss_function(logits, labels)

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

        predictions = torch.argmax(logits, dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_records += batch_size

    average_loss = total_loss / max(total_records, 1)
    accuracy = total_correct / max(total_records, 1)

    return average_loss, accuracy


# ============================================================
# 8. PREDICTION
# ============================================================

def predict(model, input_tensor):

    dataset = TensorDataset(input_tensor)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model.eval()

    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for (input_ids,) in loader:

            input_ids = input_ids.to(DEVICE)

            logits = model(input_ids)

            probabilities = torch.softmax(logits, dim=1)

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    return (
        np.array(all_predictions),
        np.array(all_probabilities)
    )


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("BYTEGUARD-IoV: TEMPORAL CLASSIFIER FINE-TUNING")
    print("=" * 75)

    print(f"Device          : {DEVICE}")
    print(f"Epochs          : {EPOCHS}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Learning rate   : {LEARNING_RATE}")
    print(f"Early stopping  : {PATIENCE} epochs")

    if not PRETRAINED_PATH.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint missing: {PRETRAINED_PATH}"
        )

    # --------------------------------------------------------
    # Load vocabulary and pretrained checkpoint
    # --------------------------------------------------------

    print("\nLoading temporal vocabulary...")

    token_to_id = load_vocabulary()

    pad_id = token_to_id["[PAD]"]

    print("\nLoading pretrained MLM checkpoint...")

    try:
        checkpoint = torch.load(
            PRETRAINED_PATH,
            map_location=DEVICE,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(
            PRETRAINED_PATH,
            map_location=DEVICE
        )

    model_config = checkpoint["model_config"]

    if model_config["vocab_size"] != len(token_to_id):
        raise ValueError("Checkpoint and vocabulary sizes differ.")

    if checkpoint["vocabulary"] != token_to_id:
        raise ValueError(
            "Checkpoint vocabulary does not match temporal vocab.json."
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading temporal datasets...")

    train_x, train_y, train_df = load_dataset(
        TRAIN_PATH,
        token_to_id
    )

    validation_x, validation_y, validation_df = load_dataset(
        VALIDATION_PATH,
        token_to_id
    )

    test_x, test_y, test_df = load_dataset(
        TEST_PATH,
        token_to_id
    )

    followup_x, followup_df = load_dataset(
        FOLLOWUP_PATH,
        token_to_id,
        require_labels=False
    )

    # The follow-up dataset must contain normal records only.
    if "Class_Label" not in followup_df.columns:
        raise ValueError("Follow-up dataset has no Class_Label column.")

    if not (followup_df["Class_Label"].astype(int) == 0).all():
        raise ValueError(
            "Follow-up dataset contains non-normal records."
        )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        TensorDataset(validation_x, validation_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Initialize model and load pretrained weights
    # --------------------------------------------------------

    print("\nInitializing classifier...")

    encoder = ByteGuardBERT(
        vocab_size=model_config["vocab_size"],
        pad_id=model_config["pad_id"],
        embed_dim=model_config["embed_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ffn_dim=model_config["ffn_dim"],
        max_seq_len=model_config["max_seq_len"],
        dropout=model_config["dropout"]
    )

    encoder.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    print("Pretrained encoder weights loaded successfully.")

    model = ByteGuardClassifier(
        encoder=encoder,
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Trainable parameters: {total_parameters:,}")

    # --------------------------------------------------------
    # Optimizer and loss
    # --------------------------------------------------------

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Fine-tuning
    # --------------------------------------------------------

    history = {
        "model": "ByteGuard-IoV Temporal Classifier",
        "pretrained_checkpoint": str(PRETRAINED_PATH),
        "architecture": model_config,
        "training_configuration": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "early_stopping_patience": PATIENCE,
            "random_seed": RANDOM_SEED
        },
        "class_names": CLASS_NAMES,
        "epochs": []
    }

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    print("\n" + "=" * 75)
    print("STARTING SUPERVISED FINE-TUNING")
    print("=" * 75)

    for epoch in range(1, EPOCHS + 1):

        start_time = time.time()

        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            loss_function,
            optimizer
        )

        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            loss_function
        )

        epoch_time = time.time() - start_time

        epoch_result = {
            "epoch": epoch,
            "training_loss": train_loss,
            "training_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "time_seconds": epoch_time
        }

        history["epochs"].append(epoch_result)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 55)
        print(f"Training loss       : {train_loss:.6f}")
        print(f"Training accuracy   : {train_accuracy:.6f}")
        print(f"Validation loss     : {validation_loss:.6f}")
        print(f"Validation accuracy : {validation_accuracy:.6f}")
        print(f"Time                : {epoch_time:.2f} seconds")

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "class_names": CLASS_NAMES,
                    "best_validation_loss": best_validation_loss,
                    "best_epoch": best_epoch,
                    "vocabulary": token_to_id
                },
                CLASSIFIER_PATH
            )

            print("Best classifier checkpoint saved.")

        else:

            epochs_without_improvement += 1

            print(
                "No validation improvement: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping activated.")
            break

    history["best_epoch"] = best_epoch
    history["best_validation_loss"] = best_validation_loss
    history["checkpoint"] = str(CLASSIFIER_PATH)

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, allow_nan=False)

    # --------------------------------------------------------
    # Load best checkpoint for final evaluation
    # --------------------------------------------------------

    print("\nLoading best validation checkpoint...")

    try:
        best_checkpoint = torch.load(
            CLASSIFIER_PATH,
            map_location=DEVICE,
            weights_only=False
        )
    except TypeError:
        best_checkpoint = torch.load(
            CLASSIFIER_PATH,
            map_location=DEVICE
        )

    model.load_state_dict(
        best_checkpoint["model_state_dict"],
        strict=True
    )

    # --------------------------------------------------------
    # Test-set evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("TEMPORAL TEST-SET EVALUATION")
    print("=" * 75)

    test_predictions, test_probabilities = predict(
        model,
        test_x
    )

    test_true = test_y.numpy()

    accuracy = accuracy_score(
        test_true,
        test_predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_true,
        test_predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    report = classification_report(
        test_true,
        test_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        test_true,
        test_predictions,
        labels=list(range(NUM_CLASSES))
    )

    print(f"Test accuracy : {accuracy:.6f}")
    print(f"Macro precision: {precision:.6f}")
    print(f"Macro recall   : {recall:.6f}")
    print(f"Macro F1-score : {f1:.6f}")

    print("\nClassification report:")

    print(
        classification_report(
            test_true,
            test_predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    report_output = {
        "evaluation": "Temporal test set",
        "number_of_records": len(test_true),
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": CLASS_NAMES,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report_output, file, indent=4, allow_nan=False)

    pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    ).to_csv(CONFUSION_PATH)

    test_output = test_df.copy()

    test_output["True_Label"] = test_true
    test_output["Predicted_Label"] = test_predictions
    test_output["True_Class"] = [
        CLASS_NAMES[value] for value in test_true
    ]
    test_output["Predicted_Class"] = [
        CLASS_NAMES[value] for value in test_predictions
    ]
    test_output["Correct"] = (
        test_true == test_predictions
    )

    for class_id, class_name in enumerate(CLASS_NAMES):
        test_output[f"Probability_{class_name}"] = (
            test_probabilities[:, class_id]
        )

    test_output.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Normal-only follow-up evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("NORMAL-ONLY FOLLOW-UP EVALUATION")
    print("=" * 75)

    followup_predictions, followup_probabilities = predict(
        model,
        followup_x
    )

    # All follow-up records are known normal.
    followup_true = np.zeros(
        len(followup_predictions),
        dtype=int
    )

    false_positives = int(
        np.sum(followup_predictions != 0)
    )

    true_negatives = int(
        np.sum(followup_predictions == 0)
    )

    false_positive_rate = (
        false_positives / len(followup_predictions)
        if len(followup_predictions) > 0
        else 0.0
    )

    normal_acceptance_rate = (
        true_negatives / len(followup_predictions)
        if len(followup_predictions) > 0
        else 0.0
    )

    predicted_distribution = {
        CLASS_NAMES[class_id]: int(
            np.sum(followup_predictions == class_id)
        )
        for class_id in range(NUM_CLASSES)
    }

    followup_report = {
        "evaluation": "Normal-only chronological follow-up",
        "number_of_records": len(followup_predictions),
        "true_normal_records": len(followup_predictions),
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_positive_rate": float(false_positive_rate),
        "normal_acceptance_rate": float(normal_acceptance_rate),
        "predicted_class_distribution": predicted_distribution,
        "interpretation": (
            "This dataset contains normal records only. "
            "The false positive rate measures the proportion "
            "of normal records classified as an attack. "
            "It does not measure attack detection performance."
        )
    }

    print(f"Follow-up records       : {len(followup_predictions):,}")
    print(f"Correctly accepted      : {true_negatives:,}")
    print(f"False positives         : {false_positives:,}")
    print(f"False positive rate     : {false_positive_rate:.6f}")
    print(f"Normal acceptance rate  : {normal_acceptance_rate:.6f}")

    print("\nPredicted class distribution:")

    for class_name, count in predicted_distribution.items():
        print(f"{class_name:<20}: {count:,}")

    with open(
        FOLLOWUP_REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            followup_report,
            file,
            indent=4,
            allow_nan=False
        )

    followup_output = followup_df.copy()

    followup_output["True_Label"] = followup_true
    followup_output["Predicted_Label"] = followup_predictions
    followup_output["Predicted_Class"] = [
        CLASS_NAMES[value] for value in followup_predictions
    ]
    followup_output["False_Positive"] = (
        followup_predictions != 0
    )

    for class_id, class_name in enumerate(CLASS_NAMES):
        followup_output[f"Probability_{class_name}"] = (
            followup_probabilities[:, class_id]
        )

    followup_output.to_csv(
        FOLLOWUP_PREDICTIONS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("TEMPORAL CLASSIFICATION COMPLETED")
    print("=" * 75)

    print(f"Best epoch             : {best_epoch}")
    print(f"Best validation loss   : {best_validation_loss:.6f}")
    print(f"Test accuracy          : {accuracy:.6f}")
    print(f"Test macro F1          : {f1:.6f}")
    print(f"Follow-up false positive rate: {false_positive_rate:.6f}")

    print("\nGenerated files:")
    print(f"Classifier checkpoint : {CLASSIFIER_PATH}")
    print(f"Training history      : {HISTORY_PATH}")
    print(f"Test report           : {REPORT_PATH}")
    print(f"Confusion matrix      : {CONFUSION_PATH}")
    print(f"Follow-up report      : {FOLLOWUP_REPORT_PATH}")
    print(f"Test predictions      : {TEST_PREDICTIONS_PATH}")
    print(f"Follow-up predictions : {FOLLOWUP_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
