# NU Arabic ABSA Improvement Plan (2026-04-24)

## 1. Scope and Method

This plan is based on:

- Static review of core modules in training, inference, evaluation, data contracts, and taxonomy.
- Runtime checks:
  - `uv run pytest -q`
  - `uv run python main.py evaluate --predictions-json predictions.json --validation-csv data/DeepX_validation.csv`
  - timed inference on validation (`/usr/bin/time ... main.py predict ...`)
- Slice-level error analysis for:
  - `predictions.json`
  - `predictions_after_semisupervised.json`
- Hot-path profiling for normalization and sentiment inference feature usage.

## 2. Current Baseline

### 2.1 Functional Health

- Test suite currently fails during collection.
- Root cause: tests import a missing constant name (`ASPECTS`) from taxonomy.

### 2.2 Metric Baseline

From `predictions.json` against validation:

- Aspect detection micro-F1: 0.7835
- Aspect detection macro-F1: 0.6957
- Sentiment macro-F1 given overlap: 0.6382
- Sentiment coverage over gold aspects: 0.7667
- End-to-end tuple F1: 0.6861

From `predictions_after_semisupervised.json` against validation:

- Aspect detection micro-F1: 0.7848
- Sentiment macro-F1 given overlap: 0.7174
- Sentiment coverage over gold aspects: 0.8250
- End-to-end tuple F1: 0.6954

### 2.3 Slice Pattern Summary

- Weakest length bucket is short reviews (`0-80`) in both prediction sets.
- Single-aspect reviews are harder than multi-aspect (`3+`) reviews.
- Category-level instability exists in low-support categories (some categories drop to near-zero tuple F1).

## 3. Key Findings

## P0 (Critical, fix first)

1. Taxonomy API inconsistency breaks tests and contract reliability.
- Impact: CI unusable; hidden regressions likely.
- Evidence:
  - `tests/test_models_smoke.py` and `tests/test_taxonomy_and_schemas.py` import `ASPECTS`.
  - `src/absa/config/taxonomy.py` exports `ASPECT_TAXONOMY`, not `ASPECTS`.

2. Predict CLI ignores `--model-dir` argument.
- Impact: misleading behavior, deploy-time model selection risk.
- Evidence:
  - argument defined in `main.py` but not consumed.
  - hard-coded loads use `artifacts/...` paths.

3. Calibration fallback mismatch between training and inference defaults.
- Impact: inconsistent behavior between offline threshold optimization and online selection.
- Evidence:
  - `src/absa/training/calibrate.py`: `apply_thresholds(... fallback_none_threshold=0.3)`
  - same file: `save_thresholds_config(... fallback_none_threshold=0.6)`

## P1 (High)

4. Sentiment model dense conversion is a major memory bottleneck.
- Impact: high RAM use, reduced scalability, possible OOM on larger batches.
- Evidence:
  - `src/absa/models/sentiment_transformer.py` calls `.toarray()` in both training and inference.
  - Estimated dense memory:
    - train sentiment instances: 3333 x 55760 -> ~1417.91 MB
    - validation sentiment instances: 840 x 55760 -> ~357.35 MB

5. Inference throughput bottleneck in end-to-end predict path.
- Impact: latency pressure for batch jobs.
- Evidence:
  - Timed command on 500-row validation: wall time ~8.71s, max RSS ~802140 KB.
  - Sentiment inference on 840 conditioned instances: ~638.9 ms total, but dense path inflates memory and copy overhead.

6. Heuristic noise and dead-path risk in aspect selection logic.
- Impact: maintainability and behavior predictability issues.
- Evidence:
  - `src/absa/inference/predict.py` defines `_is_mixed_sentiment` with noisy token list and computes `is_mixed` but does not use it in decision logic.

## P2 (Medium)

7. Main prediction flow uses labeled loader in predict command.
- Impact: unnecessary coupling to labeled schema for inference inputs.
- Evidence:
  - `main.py` predict path calls `load_labeled_reviews` instead of unlabeled-safe loading path.

8. Architecture naming drift creates confusion.
- Impact: onboarding and experimentation friction.
- Evidence:
  - `AspectTransformerModel` is currently a sparse linear surrogate, not a transformer encoder.

9. Test coverage gaps on hard ABSA constraints and serialization behavior.
- Impact: high regression probability on rules like `none` exclusivity.

## 4. Bottlenecks and Improvement Targets

### 4.1 Primary Bottlenecks

- Dense sparse-matrix materialization in sentiment model (`.toarray()`).
- Repeated Python loops in per-review inference logic.
- Contract mismatches in CLI and calibration defaults.

### 4.2 Quantitative Targets (next 2 iterations)

- Reduce predict command wall time on validation (500 rows) by at least 35%.
- Reduce peak RSS during prediction by at least 40%.
- Keep or improve tuple F1 to >= 0.695 while improving reliability.
- Raise sentiment coverage over gold aspects to >= 0.84 without reducing sentiment macro-F1 below 0.71.
- Achieve stable green tests in CI with >= 90% pass rate in new rule-focused tests.

## 5. Implementation Plan

## Phase 1 (Stabilize Contracts and Reliability) - 1 to 2 days

1. Restore taxonomy compatibility layer.
- Add `ASPECTS = ASPECT_TAXONOMY` alias in taxonomy module (or update tests and all imports consistently).
- Ensure one canonical export strategy and document it.

2. Fix `main.py` predict argument wiring.
- Route all artifact loads through `args.model_dir`.
- Add a failing-fast check when expected model files are absent.

3. Unify calibration fallback configuration.
- Define fallback thresholds only once in settings/config.
- Make training writer and inference reader consume the same fallback values.

4. Add guardrail tests.
- Add tests for:
  - taxonomy alias and ordering,
  - CLI `--model-dir` behavior,
  - calibration config roundtrip.

Exit criteria:
- `uv run pytest -q` passes.
- predict command works with non-default artifact directory.

## Phase 2 (Performance Refactor in Sentiment Path) - 2 to 4 days

1. Remove dense conversion in sentiment model.
- Replace MLP dense-tensor pipeline with sparse-friendly model path (for example linear classifier with sparse input), or introduce chunked sparse-to-dense minibatching only when truly needed.
- Keep model API stable (`fit`, `predict`, `predict_many`, `predict_proba`).

2. Batch and cache improvements.
- Avoid repeated normalization and repeated feature transforms where possible.
- Perform batched sentiment calls in predictor instead of per-aspect scalar calls when multiple aspects exist.

3. Add performance regression tests.
- Add benchmark script with thresholds for wall time and memory on validation set.

Exit criteria:
- Validation prediction max RSS reduced by >= 40%.
- Validation prediction wall time reduced by >= 35%.
- No tuple F1 regression > 0.01 absolute.

## Phase 3 (Quality Uplift and Error-Driven Modeling) - 3 to 5 days

1. Target hard slices.
- Improve short-review handling (`0-80` bucket) with specialized decision logic or threshold tuning.
- Add category-robust thresholding with minimum-support safeguards.

2. Neutral and none handling improvements.
- Improve neutral sentiment recall through class-weight and calibration tuning.
- Tighten `none` decision policy to reduce false positives in mixed-aspect contexts.

3. Formal error-analysis report generation.
- Persist standard slice reports and top failures to `artifacts/reports` per run.
- Track improvements over baseline in a run-comparison table.

Exit criteria:
- Short-bucket tuple F1 improvement >= 0.03 absolute.
- Neutral class F1 improvement >= 0.08 absolute.
- Tuple F1 >= 0.705 on validation with reproducible artifact config.

## Phase 4 (Architecture Clarity and Maintainability) - 1 to 2 days

1. Rename/model clarity cleanup.
- Either rename `AspectTransformerModel` to reflect actual implementation, or replace with true transformer path and keep naming.

2. Documentation synchronization.
- Update README and architecture doc sections where implementation intentionally diverges.

3. Operational checks.
- Add a lightweight release checklist script:
  - run tests,
  - run evaluation,
  - validate output schema,
  - print key metrics and drift from previous run.

Exit criteria:
- No naming ambiguity in model classes.
- Documentation and implementation are aligned.

## 6. Proposed Test Additions

Priority tests to add immediately:

1. `tests/test_predict_cli_model_dir.py`
- Verifies model artifacts are loaded from provided `--model-dir`.

2. `tests/test_postprocess_constraints.py`
- Verifies `none` exclusivity and `none -> neutral` invariant in finalized predictions.

3. `tests/test_calibration_config_consistency.py`
- Verifies fallback thresholds written and consumed consistently.

4. `tests/test_sentiment_serialization_roundtrip.py`
- Verifies save/load/predict behavior across normal and edge label distributions.

5. `tests/test_error_analysis_reports.py`
- Verifies slice report outputs and top failure extraction are stable and non-empty.

## 7. Execution Order and Ownership

Recommended sequence:

1. P0 contract fixes and test green state.
2. Sentiment memory/perf refactor with benchmark gate.
3. Slice-targeted model improvements and recalibration.
4. Naming/docs alignment and release automation.

Suggested ownership split:

- Engineer A: contracts, CLI wiring, tests.
- Engineer B: sentiment refactor and benchmarks.
- Engineer C: calibration and slice quality improvements.

## 8. Risk Register

- Risk: performance refactor hurts model quality.
- Mitigation: enforce tuple F1 non-regression gate.

- Risk: calibration updates change competition behavior unexpectedly.
- Mitigation: versioned threshold configs and A/B run comparison.

- Risk: architecture rename causes import breaks.
- Mitigation: temporary compatibility aliases + deprecation notices.

## 9. Verification Checklist (Definition of Done)

- Tests pass locally with `uv run pytest -q`.
- `main.py predict` respects `--model-dir`.
- Calibration defaults are consistent across train/infer.
- Validation metrics meet or exceed phase targets.
- Error-analysis artifact bundle generated and reviewed.
- README/docs updated to match shipped behavior.
