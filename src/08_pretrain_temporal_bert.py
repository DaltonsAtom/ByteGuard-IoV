
"""
BYTEGUARD-IoV
Script 08: Temporal BERT Masked Language Model Pretraining

Inputs:
    data/temporal/train.csv
    data/temporal/validation.csv
    data/temporal/vocab.json

Outputs:
    models/temporal_byteguard_mlm.pt
    outputs/temporal_metrics/temporal_pretraining_history.json

Architecture:
    Embedding dimension : 64
    Transformer layers  : 2
    Attention heads     : 4
    Feed-forward size   : 128
    Maximum sequence    : 12
    MLM probability     : 15%
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


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "temporal" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "temporal" / "validation.csv"
VOCAB_PATH = PROJECT_ROOT / "data" / "temporal" / "vocab.json"

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "temporal_metrics"

CHECKPOINT_PATH = MODEL_DIR / "temporal_byteguard_mlm.pt"
HISTORY_PATH = OUTPUT_DIR / "temporal_pretraining_history.json"

# Model configuration
EMBED_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 4
FFN_DIM = 128
MAX_SEQ_LEN = 12
DROPOUT = 0.1

# Training configuration
EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
MLM_PROBABILITY = 0.15

RANDOM_SEED = 42

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

def load_vocabulary(path):
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        vocab_data = json.load(file)

    token_to_id = vocab_data["token_to_id"]

    required_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    for token in required_tokens:
        if token not in token_to_id:
            raise ValueError(f"Missing special token: {token}")

    special_ids = {
        token_to_id[token]
        for token in required_tokens
    }

    vocab_size = len(token_to_id)

    if vocab_data.get("vocab_size", vocab_size) != vocab_size:
        raise ValueError("Vocabulary size does not match token_to_id.")

    random_token_ids = [
        token_id
        for token, token_id in token_to_id.items()
        if token not in required_tokens
    ]

    if not random_token_ids:
        raise ValueError("No ordinary tokens available for random masking.")

    print(f"Vocabulary size       : {vocab_size}")
    print(f"Special tokens        : {len(special_ids)}")
    print(f"Random replacement IDs: {len(random_token_ids)}")

    return token_to_id, vocab_size, special_ids, random_token_ids


# ============================================================
# 4. TOKENIZE BYTE SENTENCES
# ============================================================

def encode_sentences(csv_path, token_to_id):
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Byte_Sentence" not in df.columns:
        raise ValueError(
            f"Byte_Sentence column missing in {csv_path.name}"
        )

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

        # Add classification and separator tokens.
        token_ids = [cls_id] + token_ids[:MAX_SEQ_LEN - 2] + [sep_id]

        # Pad to fixed sequence length.
        token_ids += [pad_id] * (MAX_SEQ_LEN - len(token_ids))

        encoded_sequences.append(token_ids)

    tensor = torch.tensor(encoded_sequences, dtype=torch.long)

    print(f"{csv_path.name:<25}: {len(tensor):,} sequences")

    return tensor


# ============================================================
# 5. MASKED LANGUAGE MODELLING
# ============================================================

def apply_mlm_masking(
    input_ids,
    pad_id,
    mask_id,
    special_ids,
    random_token_ids,
    generator,
    probability=0.15
):
    """
    Apply BERT-style 80/10/10 masking.

    80% -> Replace with [MASK]
    10% -> Replace with random ordinary token
    10% -> Keep original token

    Labels are calculated only for selected positions.
    All other positions receive -100 and are ignored by
    CrossEntropyLoss.
    """

    masked_input = input_ids.clone()

    labels = torch.full_like(input_ids, -100)

    random_ids_tensor = torch.tensor(
        random_token_ids,
        dtype=torch.long
    )

    for row in range(input_ids.size(0)):

        sequence = input_ids[row]

        eligible_positions = torch.tensor(
            [
                position
                for position, token_id in enumerate(sequence.tolist())
                if token_id != pad_id and token_id not in special_ids
            ],
            dtype=torch.long
        )

        if len(eligible_positions) == 0:
            continue

        # Select 15% of eligible tokens.
        selection = torch.rand(
            len(eligible_positions),
            generator=generator
        ) < probability

        # Ensure at least one token is selected.
        if not selection.any():
            selected_index = torch.randint(
                0,
                len(eligible_positions),
                (1,),
                generator=generator
            )
            selection[selected_index] = True

        selected_positions = eligible_positions[selection]

        original_tokens = sequence[selected_positions]

        labels[row, selected_positions] = original_tokens

        replacement_probabilities = torch.rand(
            len(selected_positions),
            generator=generator
        )

        # 80% replacement with [MASK].
        mask_positions = replacement_probabilities < 0.80

        masked_input[
            row,
            selected_positions[mask_positions]
        ] = mask_id

        # 10% replacement with random ordinary tokens.
        random_positions = (
            (replacement_probabilities >= 0.80)
            & (replacement_probabilities < 0.90)
        )

        number_random = int(random_positions.sum().item())

        if number_random > 0:
            random_indices = torch.randint(
                0,
                len(random_ids_tensor),
                (number_random,),
                generator=generator
            )

            random_replacements = random_ids_tensor[random_indices]

            masked_input[
                row,
                selected_positions[random_positions]
            ] = random_replacements

        # Remaining 10% are intentionally unchanged.

    return masked_input, labels


# ============================================================
# 6. BYTEGUARD TRANSFORMER MODEL
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

        # MLM prediction head.
        self.mlm_dense = nn.Linear(embed_dim, embed_dim)
        self.mlm_activation = nn.GELU()
        self.mlm_norm = nn.LayerNorm(embed_dim)

        self.mlm_decoder = nn.Linear(
            embed_dim,
            vocab_size,
            bias=False
        )

        # Tie decoder weights to token embeddings.
        self.mlm_decoder.weight = self.token_embedding.weight

        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))

        self.pad_id = pad_id
        self.max_seq_len = max_seq_len

    def forward(self, input_ids):

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

        mlm_output = self.mlm_dense(hidden_states)
        mlm_output = self.mlm_activation(mlm_output)
        mlm_output = self.mlm_norm(mlm_output)

        logits = self.mlm_decoder(mlm_output) + self.mlm_bias

        return logits


# ============================================================
# 7. TRAINING AND VALIDATION
# ============================================================

def run_epoch(
    model,
    dataloader,
    optimizer,
    loss_function,
    pad_id,
    mask_id,
    special_ids,
    random_token_ids,
    training,
    epoch_seed
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_masked_tokens = 0

    generator = torch.Generator(device="cpu")
    generator.manual_seed(epoch_seed)

    for (input_ids,) in dataloader:

        input_ids = input_ids.to(DEVICE)

        masked_inputs, labels = apply_mlm_masking(
            input_ids=input_ids.cpu(),
            pad_id=pad_id,
            mask_id=mask_id,
            special_ids=special_ids,
            random_token_ids=random_token_ids,
            generator=generator,
            probability=MLM_PROBABILITY
        )

        masked_inputs = masked_inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        masked_count = int((labels != -100).sum().item())

        if masked_count == 0:
            continue

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            logits = model(masked_inputs)

            loss = loss_function(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

        total_loss += loss.item() * masked_count
        total_masked_tokens += masked_count

    average_loss = total_loss / max(total_masked_tokens, 1)

    return average_loss, total_masked_tokens


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def main():

    set_seed(RANDOM_SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("BYTEGUARD-IoV: TEMPORAL BERT PRETRAINING")
    print("=" * 75)

    print(f"Device               : {DEVICE}")
    print(f"Epochs               : {EPOCHS}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Learning rate        : {LEARNING_RATE}")
    print(f"MLM probability      : {MLM_PROBABILITY}")
    print(f"Maximum sequence len : {MAX_SEQ_LEN}")

    # --------------------------------------------------------
    # Load vocabulary
    # --------------------------------------------------------

    print("\nLoading training-only vocabulary...")

    (
        token_to_id,
        vocab_size,
        special_ids,
        random_token_ids
    ) = load_vocabulary(VOCAB_PATH)

    pad_id = token_to_id["[PAD]"]
    mask_id = token_to_id["[MASK]"]

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading temporal datasets...")

    train_tensor = encode_sentences(
        TRAIN_PATH,
        token_to_id
    )

    validation_tensor = encode_sentences(
        VALIDATION_PATH,
        token_to_id
    )

    train_dataset = TensorDataset(train_tensor)
    validation_dataset = TensorDataset(validation_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Initialize model
    # --------------------------------------------------------

    print("\nInitializing temporal ByteGuard model...")

    model = ByteGuardBERT(
        vocab_size=vocab_size,
        pad_id=pad_id,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(DEVICE)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters : {parameter_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    history = {
        "model": "ByteGuard-IoV Temporal BERT",
        "training_dataset": str(TRAIN_PATH),
        "validation_dataset": str(VALIDATION_PATH),
        "vocabulary": str(VOCAB_PATH),
        "architecture": {
            "embedding_dimension": EMBED_DIM,
            "transformer_layers": NUM_LAYERS,
            "attention_heads": NUM_HEADS,
            "feed_forward_dimension": FFN_DIM,
            "maximum_sequence_length": MAX_SEQ_LEN,
            "dropout": DROPOUT,
            "vocabulary_size": vocab_size,
            "trainable_parameters": parameter_count
        },
        "training_configuration": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "mlm_probability": MLM_PROBABILITY,
            "random_seed": RANDOM_SEED
        },
        "epochs": []
    }

    best_validation_loss = float("inf")

    print("\n" + "=" * 75)
    print("STARTING PRETRAINING")
    print("=" * 75)

    for epoch in range(1, EPOCHS + 1):

        start_time = time.time()

        train_loss, train_masked_tokens = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            pad_id=pad_id,
            mask_id=mask_id,
            special_ids=special_ids,
            random_token_ids=random_token_ids,
            training=True,
            epoch_seed=RANDOM_SEED + epoch
        )

        validation_loss, validation_masked_tokens = run_epoch(
            model=model,
            dataloader=validation_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            pad_id=pad_id,
            mask_id=mask_id,
            special_ids=special_ids,
            random_token_ids=random_token_ids,
            training=False,
            epoch_seed=RANDOM_SEED + 1000
        )

        epoch_time = time.time() - start_time

        epoch_result = {
            "epoch": epoch,
            "training_loss": train_loss,
            "validation_loss": validation_loss,
            "training_masked_tokens": train_masked_tokens,
            "validation_masked_tokens": validation_masked_tokens,
            "time_seconds": epoch_time
        }

        history["epochs"].append(epoch_result)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 55)
        print(f"Training loss          : {train_loss:.6f}")
        print(f"Validation loss        : {validation_loss:.6f}")
        print(f"Training masked tokens : {train_masked_tokens:,}")
        print(f"Validation masked tokens: {validation_masked_tokens:,}")
        print(f"Time                   : {epoch_time:.2f} seconds")

        # Save the best checkpoint based on validation loss.
        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "vocab_size": vocab_size,
                    "pad_id": pad_id,
                    "embed_dim": EMBED_DIM,
                    "num_layers": NUM_LAYERS,
                    "num_heads": NUM_HEADS,
                    "ffn_dim": FFN_DIM,
                    "max_seq_len": MAX_SEQ_LEN,
                    "dropout": DROPOUT
                },
                "vocabulary": token_to_id,
                "best_validation_loss": best_validation_loss,
                "epoch": epoch
            }

            torch.save(checkpoint, CHECKPOINT_PATH)

            print("Best model checkpoint saved.")

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history["best_validation_loss"] = best_validation_loss
    history["checkpoint"] = str(CHECKPOINT_PATH)

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4, allow_nan=False)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("TEMPORAL BERT PRETRAINING COMPLETED")
    print("=" * 75)

    print(f"Best validation loss : {best_validation_loss:.6f}")
    print(f"Model checkpoint     : {CHECKPOINT_PATH}")
    print(f"Training history     : {HISTORY_PATH}")

    print("\nNext stage: Temporal supervised fine-tuning (Script 09).")


if __name__ == "__main__":
    main()
