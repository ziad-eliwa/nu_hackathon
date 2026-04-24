# Improvement Implementation Results (2026-04-24)

## 1) What Was Implemented

This implementation focused only on preprocessing and model-performance improvements.

### Preprocessing and Feature Engineering

- Added normalization profiles via `ABSA_NORMALIZE_PROFILE` in `normalize_text`:
  - `conservative`
  - `balanced` (default)
  - `aggressive`
- Added stronger metadata/noise features:
  - `arabic_ratio`, `latin_ratio`, `digit_ratio`
  - `emoji_count`, `elongated_count`
  - existing length and punctuation features retained.

### Aspect Modeling

- Upgraded `AspectTransformerModel` surrogate:
  - richer feature stack: word n-grams + char n-grams + metadata
  - stronger class imbalance handling via `class_weight="balanced"`
  - configurable capacity from training CLI:
    - `--transformer-word-features`
    - `--transformer-char-features`
    - `--transformer-alpha`
- Added backward compatibility for existing tests/config with `max_features`.

### Sentiment Modeling

- Replaced old dense neural path with sparse class-balanced logistic regression while preserving public API:
  - no dense `toarray()` path in training/inference
  - word + char TF-IDF fused sparse features
  - configurable from training CLI:
    - `--word-max-features`
    - `--char-max-features`
    - `--c`
- Preserved API methods used by inference:
  - `fit`, `predict`, `predict_many`, `predict_proba`, `save`, `load`.

### Inference

- Removed dead mixed-sentiment heuristic path.
- Added batched sentiment prediction per review (`predict_many`) to reduce overhead.

### Experiment Automation and Recording

- Added full runner: `main.py run-improvements`
- Added implementation file: `src/absa/training/run_improvements.py`
- Each experiment now records:
  - config used
  - training duration
  - train metrics
  - full validation metrics
- Output artifacts:
  - `artifacts/improvements/all_experiment_results.json`
  - `artifacts/improvements/results_report.md`
  - per-experiment JSON under each experiment `reports/` folder.

## 2) Validation Status

- Test suite status: **PASS**
- Command run: `uv run pytest -q`
- Result: `10 passed`

## 3) Results For Every Improvement

All experiments were run on:
- train: `data/DeepX_train.csv`
- validation: `data/DeepX_validation.csv`

Baseline for delta: `exp_01_balanced_baseline`.

| Experiment | Normalize | Aspect Micro F1 | Aspect Macro F1 | Sentiment Macro F1 | Tuple F1 | Delta Tuple F1 | Train Sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| exp_01_balanced_baseline | balanced | 0.7920 | 0.6951 | 0.6334 | 0.6743 | +0.0000 | 40.9 |
| exp_02_aggressive_preprocess | aggressive | 0.7929 | 0.6956 | 0.6340 | 0.6755 | +0.0012 | 43.7 |
| exp_03_aggressive_high_capacity | aggressive | 0.7929 | 0.6956 | 0.6285 | 0.7031 | +0.0288 | 43.6 |
| exp_04_balanced_stronger_regularization | balanced | 0.7920 | 0.6951 | 0.6222 | 0.6994 | +0.0251 | 43.5 |

## 4) Best Achieved Configuration

- Best tuple F1: **0.7031**
- Best run: `exp_03_aggressive_high_capacity`
- Relative gain over baseline run: **+0.0288 tuple F1**

Config highlights:
- aggressive normalization profile
- higher aspect/sentiment feature capacity
- tuned regularization and alpha

## 5) Notes

- Convergence warnings were observed in some logistic runs (`max_iter` reached) but experiment outputs were successfully produced and recorded.
- The best model currently improves tuple F1 strongly versus this new baseline run, but it still does not reach the stretch target in the plan.
- All improvements requested were implemented in code and recorded as experiment outputs.
