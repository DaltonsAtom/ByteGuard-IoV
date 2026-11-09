# ByteGuard-IoV — Stage 6: Temporal Robustness Visualizations

## Included figures
1. **01_accuracy_f1_comparison.png** — Baseline and three inference-time masking conditions.
2. **02_performance_drop.png** — Absolute accuracy and macro-F1 degradation relative to baseline.
3. **03_per_class_f1_heatmap.png** — Per-class F1 under each masking condition.
4. **04_mean_confidence.png** — Mean maximum softmax probability for each condition.
5. **05_cpu_inference_latency.png** — Single-record mean/median and batched per-record forward-pass latency.
6. **06_model_footprint.png** — Trainable parameter count and classifier checkpoint size.
7. **07_cpu_throughput.png** — Batched CPU forward-pass throughput.

## Key results
- Test records: 7,500.
- Baseline: accuracy 99.9733%; macro-F1 99.9733%.
- CAN ID masked: accuracy 79.9733%; macro-F1 73.3141%.
- Payload masked: accuracy 20.0000%; macro-F1 6.6667%.
- Combined masking: accuracy 20.0000%; macro-F1 6.6667%.
- CPU single-record mean latency: 1.784 ms; median: 1.581 ms.
- Batched latency: 0.221 ms/record; throughput: 4,516.5 records/s.
- Model: 221,823 trainable parameters; checkpoint 0.8968 MB.

## Interpretation and limitations
The masking experiments modify inputs to the already-trained classifier at inference time. They measure sensitivity to the masked input positions; they are **not** equivalent to training separate CAN-ID-only or payload-only models and should not be described as causal feature ablation.

Script 10 assumes the first content token is the CAN ID and subsequent content tokens are payload tokens. The script itself states that this assumption must be verified against the actual `Byte_Sentence` construction. Therefore, interpret the masking labels as conditional on that token-layout assumption.

The very high mean softmax confidence under payload and combined masking does not imply correctness: those conditions have only 20% accuracy. This is evidence that confidence can remain high for incorrect predictions under severe input corruption.

Latency measures the neural-network forward pass only. It excludes CSV loading, tokenization, CAN acquisition and communication overhead. Results are hardware- and environment-dependent.

The evaluation uses the existing temporal test set and does not establish generalization to independent vehicles or recording sessions.
