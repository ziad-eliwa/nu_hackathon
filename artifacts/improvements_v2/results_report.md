# Performance Improvement Results

All experiments were run on the same train/validation split and evaluated with end-to-end tuple F1.

| Experiment | Normalize | Aspect Micro F1 | Aspect Macro F1 | Sentiment Macro F1 | Tuple F1 | Delta Tuple F1 | Train Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp_01_balanced_baseline | balanced | 0.7920 | 0.6951 | 0.6225 | 0.6560 | +0.0000 | 41.0 |
| exp_02_aggressive_preprocess | aggressive | 0.7929 | 0.6956 | 0.6230 | 0.6571 | +0.0011 | 45.8 |
| exp_03_aggressive_high_capacity | aggressive | 0.7929 | 0.6956 | 0.6526 | 0.7031 | +0.0471 | 47.9 |
| exp_04_balanced_stronger_regularization | balanced | 0.7920 | 0.6951 | 0.6345 | 0.6949 | +0.0389 | 45.4 |
| exp_05_sentiment_calibration | aggressive | 0.7929 | 0.6956 | 0.6526 | 0.7031 | +0.0471 | 45.6 |
| exp_06_platform_calibration | aggressive | 0.7929 | 0.6956 | 0.6526 | 0.7031 | +0.0471 | 48.0 |
| exp_07_neutral_boost | aggressive | 0.7929 | 0.6956 | 0.6635 | 0.7089 | +0.0529 | 50.2 |
| exp_08_neutral_aggressive | aggressive | 0.7929 | 0.6956 | 0.6303 | 0.7043 | +0.0483 | 46.5 |

## Best Configuration

- Experiment: exp_07_neutral_boost
- Tuple F1: 0.7089
- Sentiment Macro F1: 0.6635
- Aspect Micro F1: 0.7929