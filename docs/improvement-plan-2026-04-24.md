# NU Arabic ABSA Performance Plan (Preprocessing + Modeling Only)

## 1. Objective

Maximize end-to-end ABSA quality, with primary optimization target on tuple F1 and secondary targets on aspect micro/macro F1 and sentiment macro F1.

This plan intentionally excludes non-model concerns and focuses only on:

- preprocessing and data representation,
- model architecture,
- training strategy,
- calibration and decision policy as model-performance levers.

## 2. Baseline and Success Targets

Known baseline (validation):

- tuple F1: 0.686 to 0.695,
- aspect micro F1: about 0.784,
- sentiment macro F1: 0.638 to 0.717.

Performance goals (aggressive):

- Phase target A: tuple F1 >= 0.72,
- Phase target B: tuple F1 >= 0.75,
- stretch target: tuple F1 >= 0.78,
- sentiment macro F1 >= 0.76,
- `none` class F1 >= 0.60,
- `cleanliness` F1 >= 0.68.

## 3. High-Impact Strategy

1. Upgrade preprocessing to preserve Arabic signal while reducing noise.
2. Replace current pseudo-transformer surrogate with a true Arabic transformer stack.
3. Build a stronger hybrid ensemble (transformer + sparse lexical channel + metadata channel).
4. Use data-centric training upgrades: hard-example mining, pseudo-labeling, and weak supervision.
5. Calibrate and tune thresholds directly for tuple F1, not just per-aspect binary F1.

## 4. Preprocessing Roadmap (F1-Oriented)

## 4.1 Text Normalization 2.0

Current normalization is useful but too shallow for heavily noisy Arabic user reviews.

Planned upgrades:

- preserve negation cues explicitly (for example: "مو", "مش", "ليس", "ما"),
- robust punctuation normalization while preserving sentiment emojis and repetition intensity,
- Arabic letter normalization with reversible options for ablation,
- stronger elongated-token handling with learned caps per token length,
- optional spelling correction only in offline preprocessing experiments (never mandatory at inference).

Deliverable:

- preprocessing profiles (`conservative`, `balanced`, `aggressive`) and ablation table per profile.

## 4.2 Tokenization and Representation for Arabic + Code-Mix

Add dual tokenization streams:

- transformer tokenizer stream (AraBERT-family),
- character/subword lexical stream (char n-grams + fastText-style signals).

Add language-mix tags and noise tags per sample:

- arabic_ratio,
- latin_ratio,
- emoji_count,
- elongated_count,
- short_text_flag.

These become side features for both aspect and sentiment models.

## 4.3 Data Cleaning and Label Integrity

Performance-only cleaning actions:

- deduplicate near-duplicate reviews by text similarity threshold,
- detect contradictory labels for near-identical text and route to manual correction bucket,
- down-weight suspicious training rows via confidence-aware sample weights.

Expected gain:

- +0.8 to +2.0 tuple F1 points from noise reduction alone.

## 4.4 Class-Balance Engineering

For long-tail aspects (`none`, `cleanliness`, `delivery`):

- targeted oversampling,
- class-aware augmentation,
- focal-style reweighting.

For sentiment neutral scarcity:

- neutral-focused augmentation and weighted loss.

## 5. Modeling Roadmap (Major Changes Allowed)

## 5.1 Aspect Model: True Transformer Multi-Label Head

Replace current surrogate path with:

- backbone candidates:
  - `aubmindlab/bert-base-arabertv02`,
  - `CAMeL-Lab/bert-base-arabic-camelbert-mix`,
  - XLM-R large (if GPU budget allows).
- multi-label sigmoid head (9 labels),
- loss options:
  - weighted BCE,
  - asymmetric focal loss (recommended for imbalance),
  - optional label-smoothing on rare aspects.

Training settings:

- stratified multi-label folds,
- mixed precision,
- gradient accumulation,
- early stopping on tuple-F1-oriented proxy score.

## 5.2 Sentiment Model: Aspect-Conditioned Transformer

Upgrade sentiment to a transformer-based pair format:

- input: `[CLS] review [SEP] aspect [SEP]`,
- 3-way sentiment head,
- class-balanced loss or focal CE,
- optional confidence-aware training from pseudo labels.

Critical enhancement:

- train separate calibration per aspect for sentiment probabilities.

## 5.3 Hybrid Ensemble for Robustness

Final scoring should blend:

- transformer aspect probabilities,
- sparse lexical one-vs-rest probabilities,
- metadata-aware lightweight model outputs.

Blend optimization:

- optimize ensemble weights and per-aspect thresholds with Bayesian search targeting tuple F1.

## 5.4 Semi-Supervised Learning 2.0

Use unlabeled set more aggressively:

- teacher-student loop (2 to 3 rounds),
- dynamic confidence threshold per aspect (not global),
- keep only high-consistency pseudo labels under augmentation perturbations,
- curriculum: easy pseudo labels first, hard later.

Expected gain:

- +1.0 to +3.0 tuple F1 points if pseudo-label precision stays high.

## 5.5 Multi-Task Option (High-Risk, High-Reward)

Optional major architecture:

- single shared Arabic encoder,
- task head A: aspect multi-label,
- task head B: aspect-conditioned sentiment,
- joint training with weighted multitask objective.

Use only if independent-task pipeline plateaus below 0.75 tuple F1.

## 6. Experiment Design and Prioritization

## Sprint 1 (Fast Wins, 3 to 5 days)

- preprocessing profiles + ablations,
- stronger class weighting and threshold retuning for tuple F1,
- neutral-focused sentiment rebalance,
- `none`/`cleanliness` targeted improvement experiments.

Expected gain: +1.5 to +3.0 tuple F1 points.

## Sprint 2 (Core Model Upgrade, 5 to 8 days)

- true transformer aspect model,
- transformer sentiment model,
- calibrated per-aspect threshold and probability scaling,
- ensemble tuning.

Expected gain: +3.0 to +6.0 tuple F1 points over baseline.

## Sprint 3 (Semi-Supervised + Hard Slices, 4 to 7 days)

- teacher-student pseudo-labeling rounds,
- short-text expert heuristics integrated as model features,
- category-slice tuning with minimum-support constraints.

Expected gain: +1.0 to +3.0 tuple F1 points.

## 7. Hard-Slice Optimization Plan

Priority error slices:

- short reviews (`0-80` length),
- single-aspect reviews,
- low-support categories,
- rare aspects (`none`, `cleanliness`).

For each slice:

- train slice-aware calibration,
- report precision/recall/F1 deltas,
- block promotion unless global tuple F1 is non-regressed.

## 8. Evaluation Protocol (Performance-Only)

Primary metric:

- tuple F1 on validation.

Secondary metrics:

- aspect micro/macro/per-class F1,
- sentiment macro F1 and per-class F1,
- coverage over gold aspects.

Required reporting per run:

- overall metrics,
- per-aspect metrics,
- slice metrics by platform, length, aspect count, category,
- top 50 failure cases.

Model promotion rule:

- promote only if tuple F1 improves by >= 0.8 absolute points and no critical-class collapse (`none`, `cleanliness`, `neutral`).

## 9. Risk Controls for Major Changes

Risks:

- pseudo-label drift,
- overfitting to validation,
- rare-class collapse when optimizing micro metrics.

Controls:

- fold-level validation and variance tracking,
- conservative pseudo-label filtering with consistency checks,
- checkpoint selection by tuple F1 + rare-class floor constraints.

## 10. Definition of Done

The plan is successful when:

- preprocessing pipeline has validated ablations and selected best profile,
- transformer-first aspect and sentiment models are integrated,
- ensemble + calibration are tuple-F1-optimized,
- validation tuple F1 reaches at least 0.75 (or best achieved with full experiment log),
- rare aspects and neutral sentiment show measurable F1 uplift versus baseline.
