# Performance Improvement Results

All experiments were run on the same train/validation split and evaluated with end-to-end tuple F1.

| Experiment | Normalize | Aspect Micro F1 | Aspect Macro F1 | Sentiment Macro F1 | Tuple F1 | Delta Tuple F1 | Train Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp_01_balanced_baseline | balanced | 0.7920 | 0.6951 | 0.6334 | 0.6743 | +0.0000 | 40.9 |
| exp_02_aggressive_preprocess | aggressive | 0.7929 | 0.6956 | 0.6340 | 0.6755 | +0.0012 | 43.7 |
| exp_03_aggressive_high_capacity | aggressive | 0.7929 | 0.6956 | 0.6285 | 0.7031 | +0.0288 | 43.6 |
| exp_04_balanced_stronger_regularization | balanced | 0.7920 | 0.6951 | 0.6222 | 0.6994 | +0.0251 | 43.5 |

## Best Configuration

- Experiment: exp_03_aggressive_high_capacity
- Tuple F1: 0.7031
- Sentiment Macro F1: 0.6285
- Aspect Micro F1: 0.7929