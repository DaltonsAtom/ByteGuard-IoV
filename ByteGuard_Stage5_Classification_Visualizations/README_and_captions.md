# ByteGuard-IoV — Stage 5: Temporal Classification Visualizations

## Included figures
1. **01_training_validation_loss.png** — Cross-entropy loss across fine-tuning epochs.
2. **02_training_validation_accuracy.png** — Training and validation accuracy across epochs.
3. **03_test_confusion_matrix.png** — Five-class confusion matrix for the 7,500-record temporal test set.
4. **04_per_class_metrics.png** — Precision, recall, and F1-score by class.
5. **05_normal_followup_false_positives.png** — Correct normal acceptance and false positives on the normal-only chronological follow-up set.
6. **06_epoch_duration.png** — Fine-tuning duration by epoch; runtime is environment-dependent.

## Key reported results
- Best checkpoint: epoch 10; validation loss 0.002927.
- Temporal test set: 7,500 records; accuracy 99.9733%; macro-F1 99.9733%.
- Normal-only follow-up: 20,000 records; 20,000 correctly accepted; 0 false positives; observed FPR 0.00%.

## Interpretation and limitations
The temporal test set contains later chronological portions of the same source recordings used to construct the temporal partitions. These results therefore evaluate chronological holdout performance within those recordings; they do not establish generalization to unseen vehicles, unseen recording sessions, or different CAN environments.

The normal-only follow-up set can estimate false-positive behaviour on normal records only. It cannot measure attack recall or attack detection capability. The observed 0% false-positive rate is specific to the 20,000 records evaluated and should not be interpreted as a guarantee of zero false alarms in deployment.

The per-class chart uses a narrow vertical range to make small metric differences visible. Consult `per_class_metrics.csv` for exact values.

## Source artifacts
Generated from the temporal fine-tuning history, temporal test classification report/confusion matrix, and normal-only follow-up report produced by Script 09. Figures are descriptive visualizations; no additional model training was performed.
