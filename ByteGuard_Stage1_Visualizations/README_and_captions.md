# ByteGuard-IoV — Stage 1 Visualizations

All count-based figures were generated from the uploaded `preprocessing_report.json`.
The workflow diagram was derived from `02_preprocess_data.py`.

## Figures

1. `01_dataset_size_comparison.png` — Number of records in each source dataset.
2. `02_normal_vs_attack_counts.png` — Normal and attack record counts within each dataset.
3. `03_attack_percentage_by_dataset.png` — Attack records as a percentage of each dataset.
4. `04_overall_class_distribution.png` — Aggregate normal and attack-class counts across all four datasets.
5. `05_preprocessing_workflow.png` — Parsing, validation, labelling and byte-sentence construction workflow.

## Data notes

- Total records: 16,569,475.
- Accepted records: 16,569,475.
- Rejected records: 0.
- Normal records: 14,237,958.
- Attack records: 2,331,517.
- `Byte_Sentence` is constructed as CAN ID followed by the payload bytes declared by DLC.
- The source preprocessing report supports aggregate counts, but not CAN-ID frequency or payload-byte frequency plots. Those require record-level data, so they are intentionally not fabricated here.
