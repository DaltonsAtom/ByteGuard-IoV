# ByteGuard-IoV

## 1. Project Objective

ByteGuard-IoV is a machine-learning-based intrusion detection system for Controller Area Network (CAN) traffic in Internet of Vehicles (IoV) environments.

The project uses a Transformer-based model to classify CAN records into five categories:

- Normal
- Denial of Service (DoS)
- Fuzzy attack
- Gear spoofing
- RPM spoofing

The objective is to study whether byte-level CAN message representations and BERT-style masked language modelling (MLM) pretraining can support the detection of different CAN bus attacks. The temporal pipeline also evaluates chronological holdout performance, sensitivity to masked input fields, and CPU inference performance.

## 2. Dataset

The project uses four CAN intrusion datasets:

- `DoS_dataset.csv`
- `Fuzzy_dataset.csv`
- `gear_dataset.csv`
- `RPM_dataset.csv`

The source records contain fields such as timestamp, CAN ID, data length code (DLC), payload bytes, and a traffic flag. The `Byte_Sentence` field contains the tokenized representation used by the model, while `Class_Label` contains the target class.

The temporal preparation stage produces:

- `data/temporal/train.csv`
- `data/temporal/validation.csv`
- `data/temporal/test.csv`
- `data/temporal/normal_followup.csv`
- `data/temporal/vocab.json`

The training, validation, and test files are used for model development and evaluation. The normal-only follow-up file is used to measure false-positive behaviour on additional normal traffic.

## 3. Project Workflow

Run the scripts from the project root directory and follow the stages in order.

### Stage 1: Dataset inspection and preprocessing

Inspect the source datasets, check their fields and traffic labels, and prepare the CAN records for model input. The preprocessing stage creates the byte-level representation and class labels used by the later scripts.

### Stage 2: Initial model pipeline

The initial pipeline prepares model data, performs masked language modelling pretraining, fine-tunes the classifier, and evaluates its performance. This random-split experiment provides an initial baseline.

Random record splitting can place similar or repeated traffic patterns in different partitions. Therefore, the initial results should be interpreted separately from the later temporal evaluation.

### Stage 3: Temporal dataset preparation

Run the temporal preparation scripts in order (`07a` through `07f`).

This stage examines the chronological structure of the source recordings, creates time-based partitions, generates the temporal datasets, and audits the resulting data and vocabulary.

### Stage 4: Temporal MLM pretraining

Run:

```bash
python src/08_pretrain_temporal_bert.py
```

This script trains the Transformer encoder using masked language modelling on the temporal training data and monitors the training process using the temporal validation data.

Main outputs:

- `models/temporal_byteguard_mlm.pt`
- `outputs/temporal_metrics/temporal_pretraining_history.json`

### Stage 5: Temporal classifier fine-tuning

Run:

```bash
python src/09_finetune_temporal_classifier.py
```

This script loads the pretrained encoder and fine-tunes it to classify CAN records into the five classes. It selects the best checkpoint using validation loss, evaluates the model on the temporal test set, and evaluates it on the normal-only follow-up dataset.

Main outputs:

- `models/temporal_byteguard_classifier.pt`
- `outputs/temporal_metrics/temporal_classification_history.json`
- `outputs/temporal_metrics/temporal_test_classification_report.json`
- `outputs/temporal_metrics/temporal_test_confusion_matrix.csv`
- `outputs/temporal_metrics/temporal_followup_report.json`
- Test and follow-up prediction CSV files

### Stage 6: Robustness and performance evaluation

Run:

```bash
python src/10_robustness_evaluation.py
```

This script evaluates the trained classifier under four input conditions:

1. Baseline input
2. CAN ID masking
3. Payload masking
4. Combined CAN ID and payload masking

It also measures CPU inference latency, batched throughput, model parameter count, and checkpoint size.

Main outputs:

- `outputs/temporal_metrics/robustness_summary.json`
- `outputs/temporal_metrics/robustness_comparison.csv`
- `outputs/temporal_metrics/robustness_per_class.csv`
- `outputs/temporal_metrics/robustness_predictions.csv`
- `outputs/temporal_metrics/robustness_latency.json`

The masking experiments modify inputs to the already-trained model. They measure sensitivity to masked input positions; they are not equivalent to training separate models using only CAN ID or only payload information.

## 4. Recommended Execution Order

From the project root directory:

```bash
# Run the original dataset inspection and preprocessing scripts first.
# Then run the initial model pipeline if reproducing the baseline experiment.

# Prepare and audit the temporal datasets.
# Run scripts 07a through 07f in order.

# Pretrain the temporal Transformer.
python src/08_pretrain_temporal_bert.py

# Fine-tune and evaluate the temporal classifier.
python src/09_finetune_temporal_classifier.py

# Run robustness and inference performance evaluation.
python src/10_robustness_evaluation.py
```

Scripts 08, 09, and 10 depend on outputs from the preceding stages. Script 09 requires the MLM checkpoint generated by Script 08. Script 10 requires the classifier checkpoint generated by Script 09.

## 5. Environment and Dependencies

Use Python 3. The main libraries used by the model and evaluation scripts include:

- PyTorch
- NumPy
- Pandas
- scikit-learn
- Matplotlib (for visualizations)

If the project includes a `requirements.txt`, install the dependencies with:

```bash
pip install -r requirements.txt
```

Activate the project's virtual environment before running the scripts, if one is used.

```bash
# Windows
.venv\Scripts\activate

# Linux
source .venv/bin/activate
```

Run commands from the project root directory so that the relative data, model, and output paths resolve correctly.

## 6. Visualizations

The Stage 3 to Stage 6 visualization packages contain figures and supporting tables generated from the temporal dataset, pretraining, classification, and robustness results.

Each package includes a `README_and_captions.md` file describing its figures. The visualization files are for analysis and reporting; they do not train or modify the model.

## 7. Evaluation Notes and Limitations

- The temporal test set contains later chronological portions of the same source recordings used to construct the temporal partitions. These results do not independently establish generalization to unseen vehicles or independent recording sessions.
- The normal-only follow-up dataset measures false-positive behaviour on normal records. It does not measure attack detection recall.
- A zero observed false-positive rate applies only to the evaluated follow-up records; it is not a guarantee of zero false alarms in deployment.
- Script 10 measures neural-network forward-pass latency. It excludes data acquisition, CSV loading, tokenization, and communication overhead.
- Input masking is a sensitivity test, not a controlled feature-ablation study.
- Script 10 assumes the first content token is the CAN ID and subsequent content tokens are payload tokens. Verify this assumption against the actual `Byte_Sentence` construction before making feature-specific conclusions.

## 8. Expected Project Structure

The relevant files are expected to follow a structure similar to:

```text
ByteGuard-IoV/
├── data/
│   ├── original/
│   │   ├── DoS_dataset.csv
│   │   ├── Fuzzy_dataset.csv
│   │   ├── gear_dataset.csv
│   │   └── RPM_dataset.csv
│   └── temporal/
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       ├── normal_followup.csv
│       └── vocab.json
├── models/
│   ├── temporal_byteguard_mlm.pt
│   └── temporal_byteguard_classifier.pt
├── outputs/
│   └── temporal_metrics/
├── src/
│   ├── temporal preparation scripts (07a–07f)
│   ├── 08_pretrain_temporal_bert.py
│   ├── 09_finetune_temporal_classifier.py
│   └── 10_robustness_evaluation.py
└── README.md
```

The exact names and locations of the initial preprocessing scripts may differ between project versions.

## 9. Summary

The temporal workflow prepares and audits chronological CAN data, pretrains a Transformer using masked language modelling, fine-tunes it for five-class intrusion detection, and evaluates its input sensitivity and CPU inference performance. The generated reports, CSV files, checkpoints, and visualizations provide the outputs needed to inspect and document each stage.
