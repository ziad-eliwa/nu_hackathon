# Copilot Instructions for `NU` (Arabic ABSA Project)

## Build, test, and lint commands

This repository currently has packaging metadata in `pyproject.toml` but no committed test suite or lint config yet. Use `uv` as the default environment/tool runner.

### Environment setup

```bash
uv sync
```

### Run the project entry point

```bash
uv run python main.py
```

### Tests

There are no committed tests under `tests/` yet. Once tests are added, run:

```bash
uv run pytest
```

Run a single test (pattern expected by future sessions):

```bash
uv run pytest tests/test_<module>.py::test_<case_name>
```

### Linting

No linter configuration is committed yet (no `[tool.ruff]`, `flake8`, or similar config found). If/when Ruff is added, prefer:

```bash
uv run ruff check .
```

---

## High-level architecture

The repository is in an early scaffold state, but the intended system architecture is documented in:

- `docs/arabic-absa-architecture-and-plan.md`

Use that document as the source of truth for implementation direction. The planned system is a layered Arabic ABSA pipeline:

1. **Schema/contract layer** for strict label/output validation.
2. **Ingestion + normalization** for noisy Arabic/mixed-language reviews.
3. **Dual representation** (transformer + sparse lexical features).
4. **Multi-label aspect detection** with calibrated per-aspect thresholds.
5. **Aspect-conditioned sentiment classification** (`positive` / `negative` / `neutral`).
6. **Constraint/post-processing engine** (including strict `none -> neutral` handling).
7. **Evaluation + slicing** and **submission packaging**.

Current code is minimal (`main.py` placeholder), while datasets and architecture docs are already present:

- Labeled/validation/unlabeled data under `data/`
- Initial EDA notebook in `notebooks/01_eda.ipynb`

When implementing new code, align module boundaries and training/inference flow with the layered plan in `docs/arabic-absa-architecture-and-plan.md`.

---

## Key conventions specific to this repo

1. **Task framing is ABSA-first, not generic sentiment classification.**  
   Keep output structure aspect-centric (multi-label aspects + per-aspect sentiment), not one sentiment per review.

2. **Taxonomy is fixed and must be treated as canonical:**  
   `food`, `service`, `price`, `cleanliness`, `delivery`, `ambiance`, `app_experience`, `general`, `none`.

3. **`none` handling is strict by design:**  
   if `none` is predicted, sentiment must be `neutral` and no other aspects should be emitted for that review.

4. **Prefer `uv`-based commands for reproducible execution.**  
   Existing environment metadata (`pyproject.toml`, `uv.lock`) indicates `uv` is the expected workflow.

5. **Use documented architecture before inventing alternatives.**  
   The `docs/arabic-absa-architecture-and-plan.md` file captures cross-file decisions (data characteristics, modeling constraints, and staged implementation strategy). Read/update it when making architectural changes.

## Workflow Orchestration
### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload reasearch, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the usedr: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balance)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer