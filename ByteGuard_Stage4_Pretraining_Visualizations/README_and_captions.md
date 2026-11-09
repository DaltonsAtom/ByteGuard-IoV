# ByteGuard-IoV — Stage 4: Temporal BERT Pretraining Visualizations

These figures are generated from the uploaded `temporal_pretraining_history.json`. The uploaded Script 08 was used to verify the implementation and terminology.

## Figures
1. `01_temporal_mlm_loss.png` — Training and validation masked-language-model cross-entropy loss over the three epochs.
2. `02_masked_tokens_per_epoch.png` — Number of selected masked tokens used to calculate training and validation loss in each epoch.
3. `03_pretraining_epoch_duration.png` — Wall-clock time recorded for each training epoch.

## Verified implementation details from Script 08
- Input: temporal training and validation CSVs plus the training-only vocabulary.
- Architecture: token embeddings, position embeddings, LayerNorm, dropout, two-layer Transformer encoder, and an MLM prediction head.
- Embedding dimension: 64; attention heads: 4; feed-forward dimension: 128; maximum sequence length: 12.
- MLM selection probability: 15%; selected positions use BERT-style 80/10/10 replacement (MASK / random ordinary token / unchanged).
- Optimizer: AdamW; learning rate: 0.0002; weight decay: 0.01; batch size: 64; three epochs; seed: 42.
- The checkpoint is saved when validation loss improves.

## Interpretation
- The loss curves show whether the model learns to reconstruct masked tokens and whether validation loss follows training loss.
- The masked-token plot provides context for the loss estimates.
- Epoch duration is the runtime recorded by the script; it is environment-dependent and should not be treated as a hardware-independent benchmark.

## Recorded values
- Training loss: 6.004796 to 2.209507.
- Validation loss: 3.171078 to 1.934423.
- Best validation loss: 1.934423, at epoch 3.
- Total recorded epoch time: 231.50 seconds.
