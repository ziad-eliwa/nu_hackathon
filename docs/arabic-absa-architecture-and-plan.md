# Arabic ABSA System Architecture and Implementation Plan

## 1) Objective and hard requirements

Build an Arabic ABSA pipeline that, for each review:

1. detects all applicable aspects (multi-label),
2. predicts sentiment per detected aspect (`positive`, `negative`, `neutral`),
3. outputs strict, valid submission JSON.

Fixed aspect taxonomy:

- `food`
- `service`
- `price`
- `cleanliness`
- `delivery`
- `ambiance`
- `app_experience`
- `general`
- `none`

Non-negotiable output rule in this design:

- if `none` is predicted, sentiment is always `neutral`, and no other aspect is returned for that review.

---

## 2) Data-driven observations from this repository

Data files analyzed:

- `data/DeepX_train.csv` (1,971 labeled)
- `data/DeepX_validation.csv` (500 labeled)
- `data/DeepX_unlabeled.csv` (7,047 unlabeled)

### Label and text profile

- Combined labeled set: 2,471 reviews.
- Multi-aspect reviews are common:
  - 1 aspect: 1,339
  - 2 aspects: 685
  - 3+ aspects: 447
- Mixed polarity within the same review exists (~7.37%), so review-level sentiment shortcuts are unsafe.

### Aspect imbalance (combined train+validation)

- `service`: 1,241
- `app_experience`: 573
- `food`: 556
- `ambiance`: 478
- `price`: 434
- `general`: 377
- `cleanliness`: 236
- `delivery`: 209
- `none`: 69

### Sentiment skew by aspect (combined train+validation)

- `delivery`: heavily negative (~88% negative)
- `app_experience`: mostly negative (~69% negative)
- `price`: mostly negative (~68% negative)
- `ambiance`: mostly positive (~72% positive)
- `general`: strongly positive (~84% positive)
- `none`: always neutral

### Noise and domain diversity

- Arabic + English/French mixed samples exist.
- Emoji and elongated/noisy text are frequent.
- Very long free-form reviews are present (max length >1500 chars in labeled, >3600 in unlabeled).
- Platforms mix (`google_maps`, `play_store`) with different aspect priors.

Implication: the system should combine robust language modeling with explicit constraints and calibration.

---

## 3) Layered architecture (production-style)

## Layer L0 — Contracts and schema safety

Purpose: prevent invalid training/inference states and invalid final submissions.

Components:

- constants module for taxonomy/sentiment vocab
- typed records for inputs and predictions
- validators:
  - allowed labels only
  - no duplicate aspects
  - deterministic aspect ordering
  - `none` exclusivity
  - sentiment map key alignment with aspects list

---

## Layer L1 — Ingestion and preprocessing

Purpose: normalize noisy multilingual reviews while preserving sentiment signals.

Pipeline:

- robust CSV ingestion (quoted fields, multiline text)
- text normalization:
  - unicode normalization
  - Arabic char normalization (`أ/إ/آ -> ا`, `ى -> ي`, tatweel removal)
  - whitespace/punctuation cleanup
  - elongated character reduction (controlled)
- keep emojis and mixed-script tokens as optional cues
- metadata extraction:
  - `platform`, `business_category`, `star_rating`

Outputs:

- normalized review text
- metadata feature object
- cached preprocessed dataset artifacts

---

## Layer L2 — Dual representation layer

Purpose: improve robustness under limited hardware.

Representations:

1. Transformer tokens (AraBERT-compatible).
2. Sparse lexical features (TF-IDF char/word n-grams).

Why dual:

- transformer handles semantics and context;
- sparse channel captures typos, misspellings, short noisy phrases;
- useful fallback when transformer confidence is weak.

---

## Layer L3 — Aspect detection (multi-label)

Primary model:

- AraBERT-base classifier with 9 sigmoid outputs (one per aspect), trained with weighted BCE/focal-style weighting.

Secondary model:

- One-vs-rest linear classifier on sparse features + metadata.

Decision policy:

- calibrated per-aspect thresholds (not a single global threshold),
- optional weighted blending of primary + secondary probabilities,
- fallback rule when no aspect passes threshold:
  - choose `general` or `none` via calibrated confidence policy.

Post-rule:

- if any concrete aspect has confident probability, suppress `none`.

---

## Layer L4 — Aspect-conditioned sentiment classification

For each predicted aspect, build an aspect-conditioned input:

`[CLS] review_text [SEP] aspect_name [SEP]`

Classifier:

- 3-class head (`positive`, `negative`, `neutral`) on AraBERT encoder.

Training controls:

- class-weighting for rare neutral class,
- temperature calibration on validation probabilities.

Hard rule:

- force `none -> neutral`.

---

## Layer L5 — Constraint engine and post-processing

Responsibilities:

- deduplicate and sort aspects by taxonomy order,
- enforce schema invariants,
- ensure aspects list and sentiment map consistency,
- normalize low-confidence edge cases using deterministic rules.

Output:

- clean prediction object ready for JSON export.

---

## Layer L6 — Evaluation and error analysis

Required evaluation levels:

- Aspect detection: micro/macro F1.
- Sentiment (given aspect): macro F1.
- End-to-end tuple F1 over `(aspect, sentiment)`.

Mandatory slices:

- by `platform`
- by review length buckets
- by number of aspects per review
- by high-prior domains (`food_delivery`, `ecommerce`, restaurant-like categories)

Deliverables:

- confusion matrices,
- threshold report per aspect,
- top failure examples for iterative fixes.

---

## Layer L7 — Inference and submission packaging

Batch inference flow:

1. load model artifacts + calibration config,
2. preprocess review,
3. predict aspect probabilities,
4. choose aspects with calibrated thresholds,
5. infer sentiment for each chosen aspect,
6. apply constraints,
7. export strict JSON with schema validation gate.

Guardrail:

- fail-fast if any row violates schema (never silently coerce invalid output at submit time).

---

## 4) Modeling strategy for limited hardware and time

## Stage A: fast baselines (CPU-friendly)

- Aspect: TF-IDF + one-vs-rest logistic regression.
- Sentiment: aspect-conditioned logistic regression.

Goal:

- establish quick baseline,
- validate preprocessing and metrics pipeline,
- identify hard classes early.

## Stage B: parameter-efficient transformer upgrade

- Base encoder: `aubmindlab/bert-base-arabertv02` (or `bert-base-arabertv2`).
- Fine-tuning: LoRA/PEFT adapters only.
- Keep sequence length moderate by default (e.g., 128/192) and allow long-text fallback pass.

## Stage C: calibrated ensemble

- blend transformer + sparse model probabilities for aspects,
- tune per-aspect thresholds on validation to maximize tuple-level F1.

## Stage D: optional semi-supervised boost

- pseudo-label unlabeled data with high-confidence predictions only,
- retrain with confidence-weighted additional samples.

---

## 5) Proposed repository implementation layout

```text
src/absa/
  config/
    taxonomy.py
    settings.py
  data/
    schemas.py
    io.py
    splits.py
  preprocess/
    normalize.py
    metadata.py
  features/
    tfidf.py
  models/
    aspect_linear.py
    aspect_transformer.py
    sentiment_transformer.py
  training/
    train_aspect.py
    train_sentiment.py
    calibrate.py
  inference/
    predict.py
    postprocess.py
    submission.py
  evaluation/
    metrics.py
    error_analysis.py
```

Artifacts:

- `artifacts/aspect_model/`
- `artifacts/sentiment_model/`
- `artifacts/calibration/`
- `artifacts/reports/`

---

## 6) Detailed implementation plan

## Phase 1 — Foundation and contracts

- implement taxonomy constants and schema validators.
- implement strict parsers for labeled/unlabeled CSV files.
- build reproducible preprocessing module with configuration options.

Exit criteria:

- data load + validate command passes on all available datasets.

## Phase 2 — Baseline pipeline

- train and evaluate sparse multi-label aspect model.
- train and evaluate sparse aspect-conditioned sentiment model.
- implement end-to-end baseline predictor + JSON exporter.

Exit criteria:

- baseline end-to-end tuple metrics available and reproducible.

## Phase 3 — Transformer aspect detector

- train LoRA-based multi-label aspect model.
- run threshold calibration per aspect on validation.
- compare against baseline and blend if beneficial.

Exit criteria:

- improved macro-F1 or better robustness on long/noisy slices.

## Phase 4 — Transformer sentiment head

- train aspect-conditioned transformer sentiment classifier.
- calibrate class probabilities and neutral handling.
- integrate with aspect detector pipeline.

Exit criteria:

- improved sentiment macro-F1 and end-to-end tuple F1.

## Phase 5 — Constraint/post-processing hardening

- enforce `none` rule and all schema invariants.
- deterministic output ordering.
- add fail-fast submission validator.

Exit criteria:

- all generated predictions pass schema checks.

## Phase 6 — Error-driven refinements

- analyze worst confusion pairs.
- adjust preprocessing, thresholds, and blending weights.
- optional pseudo-label loop with confidence filtering.

Exit criteria:

- stable validation gains without overfitting obvious leakage.

## Phase 7 — Final packaging

- freeze model + calibration versions.
- produce final inference command and submission generator.
- archive metrics and config for reproducibility.

Exit criteria:

- one-command deterministic generation of valid submission file.

---

## 6.1) Two parallel implementation paths (for two teammates)

Use the same architecture, but split execution into two tracks with a stable contract between them.

## Path A — Data, Aspect Detection, and Calibration

Owner focus:

- data contracts and preprocessing
- multi-label aspect models (baseline + transformer)
- threshold calibration and analysis for aspect detection

Scope:

1. Implement foundational modules:
   - `src/absa/config/{taxonomy,settings}.py`
   - `src/absa/data/{schemas,io,splits}.py`
   - `src/absa/preprocess/{normalize,metadata}.py`
2. Build sparse baseline aspect detector:
   - `src/absa/features/tfidf.py`
   - `src/absa/models/aspect_linear.py`
3. Build transformer aspect detector:
   - `src/absa/models/aspect_transformer.py`
   - `src/absa/training/train_aspect.py`
4. Calibrate aspect probabilities and choose per-aspect thresholds:
   - `src/absa/training/calibrate.py`
5. Produce artifacts consumed by Path B:
   - `artifacts/aspect_model/...`
   - `artifacts/calibration/aspect_thresholds.json`
   - `artifacts/reports/aspect_eval.json`

Path A deliverables contract:

- Function/API contract:
  - `predict_aspect_probs(review_batch) -> Dict[review_id, Dict[aspect, probability]]`
- Calibration output:
  - per-aspect threshold map and fallback policy params
- Validation report:
  - aspect micro/macro F1 and slice metrics

## Path B — Sentiment, Inference Orchestration, and Submission

Owner focus:

- aspect-conditioned sentiment modeling
- post-processing/constraint engine
- end-to-end inference pipeline and submission packaging

Scope:

1. Implement sentiment model and trainer:
   - `src/absa/models/sentiment_transformer.py`
   - `src/absa/training/train_sentiment.py`
2. Implement post-processing and hard constraints:
   - `src/absa/inference/postprocess.py`
   - enforce `none -> neutral` and schema invariants
3. Implement inference pipeline and exporter:
   - `src/absa/inference/predict.py`
   - `src/absa/inference/submission.py`
4. Implement evaluation and error analysis:
   - `src/absa/evaluation/{metrics,error_analysis}.py`
5. Consume Path A outputs:
   - aspect probabilities + threshold config + aspect model artifact
6. Produce final outputs:
   - strict submission JSON
   - tuple-level metrics and confusion reports

Path B deliverables contract:

- Function/API contract:
  - `predict_sentiment(review_text, aspect) -> {positive|negative|neutral}`
  - `assemble_prediction(review) -> {"aspects": [...], "aspect_sentiments": {...}}`
- Validation report:
  - sentiment macro-F1 (given aspect)
  - end-to-end tuple F1

## Shared interface and handoff checklist

Both paths must align on the same canonical config:

- taxonomy order
- label vocabulary
- preprocessing configuration version
- artifact naming/versioning

Recommended shared files (owned jointly):

- `src/absa/config/taxonomy.py`
- `src/absa/config/settings.py`
- `src/absa/data/schemas.py`

Handoff checklist from Path A to Path B:

1. aspect probability API stable and tested
2. calibration JSON present with version tag
3. sample predictions for 50 validation rows shared
4. known failure slices documented

Integration checkpoint schedule (event-based, not time-based):

1. **Checkpoint 1:** after Path A baseline aspect model and Path B sentiment skeleton compile.
2. **Checkpoint 2:** after Path A calibration + Path B constraint engine are both ready.
3. **Checkpoint 3:** full end-to-end dry run on validation and schema verification.
4. **Checkpoint 4:** final model freeze and submission generation.

## Dependency view (parallel-safe order)

- Can run immediately in parallel:
  - Path A: Phases 1-3
  - Path B: sentiment model scaffolding + postprocess/evaluation scaffolding
- Requires Path A outputs:
  - Path B end-to-end integration and final submission pipeline
- Final shared stage:
  - Phase 6 (error-driven refinements) and Phase 7 (packaging)

---

## 7) MCP servers and skills installed in this environment

Installed skills (global):

- `data-scientist`
- `machine-learning`
- `mcp-builder`
- `find-skills` (pre-existing)

Installed MCP server packages (user-local):

- `@modelcontextprotocol/server-filesystem`
- `@modelcontextprotocol/server-memory`
- `@modelcontextprotocol/server-sequential-thinking`
- `@modelcontextprotocol/server-brave-search` (requires `BRAVE_API_KEY`)

Install location:

- `/home/hazemoonium/.local/mcp`

Server binaries:

- `/home/hazemoonium/.local/mcp/node_modules/.bin/mcp-server-filesystem`
- `/home/hazemoonium/.local/mcp/node_modules/.bin/mcp-server-memory`
- `/home/hazemoonium/.local/mcp/node_modules/.bin/mcp-server-sequential-thinking`
- `/home/hazemoonium/.local/mcp/node_modules/.bin/mcp-server-brave-search`

Note:

- local install prefix was used because global npm directories are not writable in this environment.
