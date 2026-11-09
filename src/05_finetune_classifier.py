
"""
ByteGuard-IoV — Script 05: Supervised BERT Fine-tuning

Objective:
    Fine-tune the pre-trained ByteBERT encoder for
    five-class CAN intrusion detection.

Inputs:
    data/processed/train.csv
    data/processed/validation.csv
    data/processed/test.csv
    models/byteguard_mlm.pt
    models/vocab.json

Outputs:
    models/byteguard_classifier.pt
    outputs/metrics/classification_history.json
    outputs/metrics/test_classification_report.json
    outputs/metrics/test_confusion_matrix.csv

Classes:
    0 - Normal
    1 - DoS
    2 - Fuzzy
    3 - Gear Spoofing
    4 - RPM Spoofing
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

PRETRAINED_MODEL = MODEL_DIR / "byteguard_mlm.pt"
VOCAB_FILE = MODEL_DIR / "vocab.json"

CLASSIFIER_PATH = MODEL_DIR / "byteguard_classifier.pt"

HISTORY_PATH = METRICS_DIR / "classification_history.json"
REPORT_PATH = METRICS_DIR / "test_classification_report.json"
CONFUSION_PATH = METRICS_DIR / "test_confusion_matrix.csv"

SEED = 42

MAX_LEN = 12
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FEEDFORWARD_DIM = 128

NUM_CLASSES = 5

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-4

CLASS_NAMES = [
    "Normal",
    "DoS",
    "Fuzzy",
    "Gear Spoofing",
    "RPM Spoofing",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("BYTEGUARD-IoV — SUPERVISED FINE-TUNING")
print("=" * 65)

print(f"Device: {DEVICE}")


# --------------------------------------------------
# 2. Load pretrained model and vocabulary
# --------------------------------------------------

for required_file in [
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
    PRETRAINED_MODEL,
    VOCAB_FILE,
]:
    if not required_file.exists():
        raise FileNotFoundError(
            f"Required file missing: {required_file}"
        )

with VOCAB_FILE.open("r", encoding="utf-8") as file:
    vocab = json.load(file)

checkpoint = torch.load(
    PRETRAINED_MODEL,
    map_location="cpu",
    weights_only=True,
)

PAD_ID = vocab["[PAD]"]
UNK_ID = vocab["[UNK]"]
CLS_ID = vocab["[CLS]"]
SEP_ID = vocab["[SEP]"]

# Verify that the vocabulary and pretrained checkpoint match.
assert len(vocab) == checkpoint["vocab_size"]
assert checkpoint["max_len"] == MAX_LEN
assert checkpoint["embed_dim"] == EMBED_DIM
assert checkpoint["num_heads"] == NUM_HEADS
assert checkpoint["num_layers"] == NUM_LAYERS

print(f"Vocabulary size: {len(vocab):,}")
print("Pretrained checkpoint loaded successfully.")


# --------------------------------------------------
# 3. Load the labelled datasets
# --------------------------------------------------

def load_dataset(path):
    frame = pd.read_csv(
        path,
        usecols=["Byte_Sentence", "Class_Label"],
    )

    frame["Byte_Sentence"] = (
        frame["Byte_Sentence"]
        .fillna("")
        .astype(str)
    )

    frame["Class_Label"] = pd.to_numeric(
        frame["Class_Label"],
        errors="coerce",
    )

    if frame["Class_Label"].isna().any():
        raise ValueError(
            f"Invalid Class_Label values found in {path.name}"
        )

    frame["Class_Label"] = frame["Class_Label"].astype(int)

    if not frame["Class_Label"].between(
        0, NUM_CLASSES - 1
    ).all():
        raise ValueError(
            f"Unexpected class IDs in {path.name}"
        )

    return frame


train_df = load_dataset(TRAIN_FILE)
validation_df = load_dataset(VALIDATION_FILE)
test_df = load_dataset(TEST_FILE)

print(f"\nTraining records:   {len(train_df):,}")
print(f"Validation records: {len(validation_df):,}")
print(f"Testing records:    {len(test_df):,}")


# --------------------------------------------------
# 4. Tokenization
# --------------------------------------------------

def encode_sentence(sentence):
    tokens = sentence.upper().split()

    token_ids = [CLS_ID]

    for token in tokens:
        token_ids.append(
            vocab.get(token, UNK_ID)
        )

    token_ids.append(SEP_ID)

    token_ids = token_ids[:MAX_LEN]

    attention_mask = [1] * len(token_ids)

    padding_length = MAX_LEN - len(token_ids)

    token_ids += [PAD_ID] * padding_length
    attention_mask += [0] * padding_length

    return token_ids, attention_mask


class CANIntrusionDataset(Dataset):

    def __init__(self, dataframe):
        self.sentences = dataframe[
            "Byte_Sentence"
        ].tolist()

        self.labels = dataframe[
            "Class_Label"
        ].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        token_ids, attention_mask = encode_sentence(
            self.sentences[index]
        )

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(
                self.labels[index],
                dtype=torch.long,
            ),
        )


train_dataset = CANIntrusionDataset(train_df)
validation_dataset = CANIntrusionDataset(validation_df)
test_dataset = CANIntrusionDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# --------------------------------------------------
# 5. Define classification model
# --------------------------------------------------

class ByteBERTClassifier(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
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

        # Extract the representation of the [CLS] token.
        cls_representation = hidden_states[:, 0, :]

        logits = self.classifier(
            self.dropout(cls_representation)
        )

        return logits


model = ByteBERTClassifier(
    vocab_size=len(vocab)
)

# Transfer pretrained embedding and Transformer encoder weights.
# The old MLM prediction head is deliberately excluded.
pretrained_state = checkpoint["model_state_dict"]

encoder_state = {
    key: value
    for key, value in pretrained_state.items()
    if (
        key.startswith("token_embedding.")
        or key.startswith("position_embedding.")
        or key.startswith("encoder.")
    )
}

load_result = model.load_state_dict(
    encoder_state,
    strict=False,
)

if load_result.missing_keys != [
    "classifier.weight",
    "classifier.bias",
]:
    raise RuntimeError(
        f"Unexpected missing parameters: "
        f"{load_result.missing_keys}"
    )

if load_result.unexpected_keys:
    raise RuntimeError(
        f"Unexpected pretrained parameters: "
        f"{load_result.unexpected_keys}"
    )

model = model.to(DEVICE)

print("\nPretrained encoder weights transferred.")
print("Classification head initialized.")


# --------------------------------------------------
# 6. Optimizer and loss
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01,
)


# --------------------------------------------------
# 7. Training function
# --------------------------------------------------

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_number, (
        input_ids,
        attention_mask,
        labels,
    ) in enumerate(train_loader, start=1):

        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            input_ids,
            attention_mask,
        )

        loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss encountered."
            )

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        total_loss += loss.item() * labels.size(0)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

        if batch_number % 50 == 0:
            print(
                f"  Batch {batch_number}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


# --------------------------------------------------
# 8. Validation function
# --------------------------------------------------

def evaluate(loader):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for input_ids, attention_mask, labels in loader:

            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(
                input_ids,
                attention_mask,
            )

            loss = criterion(logits, labels)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy_score(
            all_labels,
            all_predictions,
        ),
        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
    }


# --------------------------------------------------
# 9. Fine-tune and select the best model
# --------------------------------------------------

history = []

best_validation_f1 = -1.0

for epoch in range(1, EPOCHS + 1):

    print(f"\nEpoch {epoch}/{EPOCHS}")

    train_metrics = train_one_epoch()

    validation_metrics = evaluate(
        validation_loader
    )

    epoch_record = {
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "validation_loss": validation_metrics["loss"],
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
    }

    history.append(epoch_record)

    print(
        f"Training loss:       {train_metrics['loss']:.4f}"
    )

    print(
        f"Training accuracy:   {train_metrics['accuracy']:.4f}"
    )

    print(
        f"Validation loss:     {validation_metrics['loss']:.4f}"
    )

    print(
        f"Validation accuracy: {validation_metrics['accuracy']:.4f}"
    )

    print(
        f"Validation Macro F1: {validation_metrics['macro_f1']:.4f}"
    )

    # Select the best checkpoint using validation Macro F1.
    if validation_metrics["macro_f1"] > best_validation_f1:

        best_validation_f1 = validation_metrics["macro_f1"]

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab_size": len(vocab),
                "max_len": MAX_LEN,
                "embed_dim": EMBED_DIM,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "feedforward_dim": FEEDFORWARD_DIM,
                "num_classes": NUM_CLASSES,
                "class_names": CLASS_NAMES,
                "pad_id": PAD_ID,
            },
            CLASSIFIER_PATH,
        )

        print("  Best model checkpoint saved.")


# --------------------------------------------------
# 10. Save training history
# --------------------------------------------------

with HISTORY_PATH.open("w", encoding="utf-8") as file:
    json.dump(history, file, indent=4)


# --------------------------------------------------
# 11. Load best model and evaluate on test data
# --------------------------------------------------

print("\nLoading best validation checkpoint...")

best_checkpoint = torch.load(
    CLASSIFIER_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(
    best_checkpoint["model_state_dict"]
)

test_metrics = evaluate(test_loader)

# Collect predictions for detailed evaluation.
model.eval()

all_test_predictions = []
all_test_labels = []

with torch.no_grad():

    for input_ids, attention_mask, labels in test_loader:

        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        logits = model(
            input_ids,
            attention_mask,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_test_predictions.extend(
            predictions.cpu().numpy()
        )

        all_test_labels.extend(
            labels.numpy()
        )


# --------------------------------------------------
# 12. Calculate final classification metrics
# --------------------------------------------------

test_report = classification_report(
    all_test_labels,
    all_test_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

test_confusion = confusion_matrix(
    all_test_labels,
    all_test_predictions,
    labels=list(range(NUM_CLASSES)),
)

summary = {
    "test_samples": len(all_test_labels),
    "accuracy": accuracy_score(
        all_test_labels,
        all_test_predictions,
    ),
    "macro_precision": precision_score(
        all_test_labels,
        all_test_predictions,
        average="macro",
        zero_division=0,
    ),
    "macro_recall": recall_score(
        all_test_labels,
        all_test_predictions,
        average="macro",
        zero_division=0,
    ),
    "macro_f1": f1_score(
        all_test_labels,
        all_test_predictions,
        average="macro",
        zero_division=0,
    ),
    "classification_report": test_report,
    "confusion_matrix": test_confusion.tolist(),
}

with REPORT_PATH.open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

confusion_df = pd.DataFrame(
    test_confusion,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)

confusion_df.to_csv(CONFUSION_PATH)


# --------------------------------------------------
# 13. Display results
# --------------------------------------------------

print("\n" + "=" * 65)
print("BYTEGUARD-IoV CLASSIFICATION RESULTS")
print("=" * 65)

print(f"Test samples: {len(all_test_labels):,}")

print(f"Accuracy:       {summary['accuracy']:.4f}")
print(f"Macro Precision: {summary['macro_precision']:.4f}")
print(f"Macro Recall:    {summary['macro_recall']:.4f}")
print(f"Macro F1-score:  {summary['macro_f1']:.4f}")

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        all_test_labels,
        all_test_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)

print("\nCONFUSION MATRIX")

print(confusion_df)

print("\n" + "=" * 65)
print("SUPERVISED FINE-TUNING COMPLETE")
print("=" * 65)

print(f"Best model:       {CLASSIFIER_PATH}")
print(f"Training history: {HISTORY_PATH}")
print(f"Test report:      {REPORT_PATH}")
print(f"Confusion matrix: {CONFUSION_PATH}")
