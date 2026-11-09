# ByteGuard-IoV — Stage 3 Visualizations

Figures are generated from the uploaded `temporal_partition_plan.json` and `temporal_dataset_report.json`, with the six uploaded 07a–07f scripts used to interpret the methodology.

## Figures
1. `01_chronological_partition_timeline.png` — Train, validation and test windows in chronological order for each source recording. Window indices are local to each recording; they are not a shared global time axis.
2. `02_partition_record_counts.png` — Total and attack record counts in each chronological partition, as reported by the partition plan.
3. `03_attack_percentage_by_partition.png` — Attack proportion within each source recording's train, validation and test windows.
4. `04_temporal_dataset_class_distribution.png` — Class balance of the generated train, validation and test samples and the normal-only follow-up.
5. `05_source_composition.png` — Contribution of each source recording to each sampled partition.

## Method notes
- The temporal-window analysis uses fixed five-second windows.
- The split is chronological over attack-containing windows, with 70% train, 15% validation and 15% test.
- The later normal-only period is kept separately as follow-up data.
- The generated balanced datasets contain 50,000 train, 7,500 validation, 7,500 test, and 20,000 normal-follow-up records.
- The sampled train, validation and test partitions have 10,000/1,500/1,500 records per class respectively.
- The normal-only follow-up contains 20,000 Normal records and is intended for false-positive evaluation, not attack recall.
- The uploaded set did not include the detailed per-window CSV reports from Script 07c or `temporal_final_audit.json`; therefore this package does not plot the full five-second window-by-window attack activity or reproduce the final vocabulary/OOV audit statistics.
- The scripts and partition plan show chronological partitioning. The four recordings have independent timestamp ranges, so their raw timestamps should not be placed on one shared absolute-time axis.
