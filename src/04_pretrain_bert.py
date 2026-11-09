
"""
ByteGuard-IoV — Script 04: Masked Byte Pre-training

Objective:
    Pre-train a lightweight BERT-style Transformer using masked
    byte-token prediction, without using intrusion labels.

Inputs:
    data/processed/train.csv
    data/processed/validation.csv

Outputs:
    models/byteguard_mlm.pt
    models/vocab.json
    outputs/metrics/pretraining_history.json

Architecture:
    Embedding dimension: 64
    Transformer layers: 2
    Attention heads: 4
    Maximum sequence length: 12
    Masking probability: 15%

This is a small, research-inspired encoder, not a reproduction
of the original paper's complete experimental model.
"""

import json
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# 1. Configuration and paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"

MODEL_PATH = MODEL_DIR / "byteguard_mlm.pt"
VOCAB_PATH = MODEL_DIR / "vocab.json"
HISTORY_PATH = METRICS_DIR / "pretraining_history.json"

SEED = 42

MAX_LEN = 12
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FEEDFORWARD_DIM = 128

BATCH_SIZE = 128
EPOCHS = 3
LEARNING_RATE = 3e-4
MASK_PROBABILITY = 0.15

# CPU is used automatically when a CUDA GPU is unavailable.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------------------------------------
# 2. Create output directories
# --------------------------------------------------

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. Load byte sentences
# --------------------------------------------------

def load_sentences(path):
    """Load only the byte sentences; attack labels are not used."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run Scripts 02 and 03 first."
        )

    frame = pd.read_csv(
        path,
        usecols=["Byte_Sentence"],
    )

    sentences = (
        frame["Byte_Sentence"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    return sentences


print(f"Training device: {DEVICE}")

train_sentences = load_sentences(TRAIN_FILE)
validation_sentences = load_sentences(VALIDATION_FILE)

print(f"Training sequences: {len(train_sentences):,}")
print(f"Validation sequences: {len(validation_sentences):,}")


# --------------------------------------------------
# 4. Build the vocabulary from training data only
# --------------------------------------------------

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
]

# Every distinct token observed in the training byte sentences
# is included in the vocabulary. This includes CAN IDs and bytes.
observed_tokens = set()

for sentence in train_sentences:
    observed_tokens.update(sentence.upper().split())

# Sort tokens for deterministic vocabulary construction.
vocabulary_tokens = SPECIAL_TOKENS + sorted(
    observed_tokens.difference(SPECIAL_TOKENS)
)

vocab = {
    token: index
    for index, token in enumerate(vocabulary_tokens)
}

PAD_ID = vocab["[PAD]"]
UNK_ID = vocab["[UNK]"]
CLS_ID = vocab["[CLS]"]
SEP_ID = vocab["[SEP]"]
MASK_ID = vocab["[MASK]"]

SPECIAL_IDS = {
    PAD_ID,
    CLS_ID,
    SEP_ID,
    MASK_ID,
}

# Exclude special tokens from random replacement candidates.
RANDOM_TOKEN_IDS = [
    index
    for token, index in vocab.items()
    if token not in SPECIAL_TOKENS
]

if not RANDOM_TOKEN_IDS:
    raise ValueError("Vocabulary contains no ordinary tokens.")

print(f"Vocabulary size: {len(vocab):,}")


# --------------------------------------------------
# 5. Convert sentences into fixed-length token IDs
# --------------------------------------------------

def encode_sentence(sentence):
    """Convert a byte sentence to IDs and add boundary tokens."""

    tokens = sentence.upper().split()

    token_ids = [CLS_ID]

    for token in tokens:
        token_ids.append(vocab.get(token, UNK_ID))

    token_ids.append(SEP_ID)

    # Truncate sequences if they exceed the model's input limit.
    token_ids = token_ids[:MAX_LEN]

    # Padding is added to make batches rectangular.
    attention_mask = [1] * len(token_ids)

    padding_length = MAX_LEN - len(token_ids)

    token_ids += [PAD_ID] * padding_length
    attention_mask += [0] * padding_length

    return token_ids, attention_mask


class ByteSentenceDataset(Dataset):
    """Dataset containing unlabelled byte-token sequences."""

    def __init__(self, sentences):
        self.examples = [
            encode_sentence(sentence)
            for sentence in sentences
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        token_ids, attention_mask = self.examples[index]

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )


train_dataset = ByteSentenceDataset(train_sentences)
validation_dataset = ByteSentenceDataset(validation_sentences)

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


# --------------------------------------------------
# 6. Implement BERT-style masked-token corruption
# --------------------------------------------------

def mask_tokens(input_ids):
    """
    Select 15% of ordinary tokens for masked-token prediction.

    Of the selected tokens:
        80% are replaced with [MASK].
        10% are replaced with a random ordinary token.
        10% are left unchanged.

    Targets are -100 for positions that do not contribute
    to the masked-token loss.
    """

    masked_inputs = input_ids.clone()

    # Ignore all positions except selected masked-token targets.
    targets = torch.full_like(input_ids, -100)

    for row_index in range(input_ids.size(0)):
        eligible_positions = [
            position
            for position in range(input_ids.size(1))
            if input_ids[row_index, position].item()
            not in SPECIAL_IDS
            and input_ids[row_index, position].item() != UNK_ID
        ]

        if not eligible_positions:
            continue

        number_to_mask = max(
            1,
            round(len(eligible_positions) * MASK_PROBABILITY),
        )

        selected_positions = random.sample(
            eligible_positions,
            min(number_to_mask, len(eligible_positions)),
        )

        for position in selected_positions:
            original_token = input_ids[row_index, position].item()

            targets[row_index, position] = original_token

            probability = random.random()

            if probability < 0.80:
                # Replace with the mask token.
                masked_inputs[row_index, position] = MASK_ID

            elif probability < 0.90:
                # Replace with a random ordinary vocabulary token.
                masked_inputs[row_index, position] = random.choice(
                    RANDOM_TOKEN_IDS
                )

            # Otherwise leave the original token unchanged.

    return masked_inputs, targets


# --------------------------------------------------
# 7. Define the lightweight Transformer encoder
# --------------------------------------------------

class ByteBERT(nn.Module):
    """Small bidirectional Transformer for CAN byte sequences."""

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

        # Predict the original token at each masked position.
        self.mlm_head = nn.Linear(
            EMBED_DIM,
            vocab_size,
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

        # Transformer expects True for padding positions.
        padding_mask = attention_mask == 0

        hidden_states = self.encoder(
            embeddings,
            src_key_padding_mask=padding_mask,
        )

        return self.mlm_head(hidden_states)


model = ByteBERT(len(vocab)).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

loss_function = nn.CrossEntropyLoss(ignore_index=-100)


# --------------------------------------------------
# 8. Train for masked-token prediction
# --------------------------------------------------

def train_one_epoch():
    """Train on the training sequences for one epoch."""

    model.train()

    total_loss = 0.0
    total_batches = 0

    for batch_number, (input_ids, attention_mask) in enumerate(
        train_loader,
        start=1,
    ):
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)

        masked_inputs, targets = mask_tokens(input_ids)

        optimizer.zero_grad(set_to_none=True)

        logits = model(masked_inputs, attention_mask)

        loss = loss_function(
            logits.reshape(-1, len(vocab)),
            targets.reshape(-1),
        )

        # Defensive check: ensure at least one target was selected.
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss encountered.")

        loss.backward()

        # Prevent unusually large gradients from destabilising training.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if batch_number % 50 == 0:
            print(
                f"  Batch {batch_number}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    return total_loss / max(total_batches, 1)


# --------------------------------------------------
# 9. Validate masked-token prediction
# --------------------------------------------------

def evaluate_validation():
    """Evaluate masked-token loss on held-out validation sequences."""

    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for input_ids, attention_mask in validation_loader:
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            # Reset the seed so validation corruption is reproducible.
            masked_inputs, targets = mask_tokens(input_ids)

            logits = model(masked_inputs, attention_mask)

            loss = loss_function(
                logits.reshape(-1, len(vocab)),
                targets.reshape(-1),
            )

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


# --------------------------------------------------
# 10. Run pre-training and record the history
# --------------------------------------------------

history = []

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")

    train_loss = train_one_epoch()
    validation_loss = evaluate_validation()

    epoch_record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
    }

    history.append(epoch_record)

    print(f"Training loss:   {train_loss:.4f}")
    print(f"Validation loss: {validation_loss:.4f}")


# --------------------------------------------------
# 11. Save model weights and vocabulary
# --------------------------------------------------

checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab_size": len(vocab),
    "max_len": MAX_LEN,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "num_layers": NUM_LAYERS,
    "feedforward_dim": FEEDFORWARD_DIM,
    "pad_id": PAD_ID,
    "unk_id": UNK_ID,
    "cls_id": CLS_ID,
    "sep_id": SEP_ID,
    "mask_id": MASK_ID,
}

torch.save(checkpoint, MODEL_PATH)

with VOCAB_PATH.open("w", encoding="utf-8") as file:
    json.dump(vocab, file, indent=2)

with HISTORY_PATH.open("w", encoding="utf-8") as file:
    json.dump(history, file, indent=2)

print("\n" + "=" * 65)
print("MASKED BYTE PRE-TRAINING COMPLETE")
print("=" * 65)

print(f"Model weights: {MODEL_PATH}")
print(f"Vocabulary:    {VOCAB_PATH}")
print(f"Training log:  {HISTORY_PATH}")

print("\nThe encoder is ready for supervised fine-tuning.")
