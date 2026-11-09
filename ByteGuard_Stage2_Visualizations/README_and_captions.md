# ByteGuard-IoV — Stage 2 Visualizations

## Included figures

1. `01_baseline_split_sizes.png` — 50,000-record random stratified split: 35,000 train, 7,500 validation, 7,500 test.
2. `02_class_distribution_by_split.png` — Balanced class counts across the three partitions. Per-class counts are 7,000/1,500/1,500, derived from the balanced 10,000-per-class sample and stratified split.
3. `03_mlm_pretraining_loss.png` — Training and validation MLM loss over three epochs.
4. `04_classifier_loss.png` — Training and validation classification loss over five epochs.
5. `05_classifier_accuracy.png` — Training and validation accuracy over five epochs.
6. `06_test_confusion_matrix.png` — Test-set confusion matrix (7,499 correct, one Fuzzy-to-Normal error).
7. `07_per_class_metrics.png` — Precision, recall and F1-score by class.
8. `08_robustness_comparison.png` — Baseline versus CAN-ID and payload masking.
9. `09_inference_latency.png` — Average neural-network forward-pass latency under each input condition.
10. `10_roc_ap_scores.png` — Per-class ROC-AUC and Average Precision scores.
11–13. Existing user-provided ROC, precision-recall and robustness plots, copied into this package.

## Key reported results

- Test records: 7,500.
- Accuracy: 99.9867%.
- Macro precision: 99.9867%.
- Macro recall: 99.9867%.
- Macro-F1: 99.9867%.
- Correct predictions: 7,499/7,500.
- One Fuzzy record was predicted as Normal.
- Baseline average forward-pass time: 0.1143 ms per record as reported by Script 06.
- Model parameters: 211,461.
- Checkpoint size: 0.8171 MB.

## Interpretation cautions

- Scripts 03–06 use a random record-level stratified split, not a chronological split.
- The high baseline scores do not establish generalisation to unseen recordings or vehicles.
- Masking experiments are inference-time sensitivity tests, not independently trained feature ablations.
- Latency is the model forward-pass measurement; it excludes CAN acquisition, tokenization, preprocessing, and communication.
- ROC-AUC and Average Precision are both 1.0 for all classes in the supplied report. The rounded values and near-perfect performance should be interpreted in light of the random split.
