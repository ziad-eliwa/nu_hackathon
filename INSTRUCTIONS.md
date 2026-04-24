# Instructions for future sessions (NU - Arabic ABSA Project)

## Quick Reference

### Environment Commands
```bash
uv sync --all-extras         # Install all dependencies (including dev)
uv run pytest               # Run tests - all 10 pass
uv run pytest tests/test_<module>.py::test_<case_name>  # Single test
```

### Project Structure
```
src/absa/
  config/       - taxonomy.py (9 aspects), settings.py
  data/        - schemas.py, io.py, splits.py
  preprocess/  - normalize.py, metadata.py
  features/    - tfidf.py
  models/      - aspect_linear.py, aspect_transformer.py, aspect_api.py
  training/    - train_aspect.py, calibrate.py
```

### Key Information
- **Task**: Arabic Aspect-Based Sentiment Analysis (multi-label aspect + per-aspect sentiment)
- **9 Aspects**: food, service, price, cleanliness, delivery, ambiance, app_experience, general, none
- **Sentiments**: positive, negative, neutral
- **Hard rule**: if `none` predicted → sentiment must be `neutral`, no other aspects
- **Datasets**: data/DeepX_train.csv (1971), DeepX_validation.csv (500), DeepX_unlabeled.csv (7047)

### Architecture
- Layered pipeline: L0 contracts → L1 ingest/normalization → L2 dual rep → L3 aspect detection → L4 sentiment → L5 constraints → L6 eval → L7 inference
- Models: AspectLinearModel (TF-IDF baseline) + AspectTransformerModel (n-gram surrogate) → blended + calibrated thresholds
- Entry point: uv run python -m absa.training.train_aspect

### Running Training
```bash
uv run python -m absa.training.train_aspect --train-csv data/DeepX_train.csv --validation-csv data/DeepX_validation.csv --artifacts-root artifacts
```

### Files to Reference
- .github/copilot-instructions.md - Detailed project conventions
- docs/arabic-absa-architecture-and-plan.md - Full architecture & phases

## Installed Skills (already loaded)
- data-scientist: ML, stats, A/B testing, experiment tracking
- machine-learning: JAX, functional ML patterns
- mcp-builder: Building MCP servers (FastMCP/Node)
- find-skills: Discovering skills

## Installed MCP Servers
Location: /home/hazemoonium/.local/mcp/
- @modelcontextprotocol/server-filesystem
- @modelcontextprotocol/server-memory  
- @modelcontextprotocol/server-sequential-thinking
- @modelcontextprotocol/server-brave-search (requires BRAVE_API_KEY)

Note: gh CLI is not installed. For GitHub Copilot MCP integration, the following packages are available and can be used via npx:

## GitHub Copilot MCP Packages (npm)
- `@microsoft/github-copilot-app-modernization-mcp-server` - Java/.NET app modernization
- `@willianpaiva/copilot-mcp-server` - Chat with Copilot AI, code explanations/suggestions/reviews
- `awesome-copilot-mcp` - Access to awesome-copilot agents and collections
- `@trishchuk/copilot-mcp-server` - GitHub Copilot CLI integration

## MCP Configuration Examples
```json
{
  "mcpServers": {
    "copilot": {
      "command": "npx",
      "args": ["-y", "@willianpaiva/copilot-mcp-server"]
    }
  }
}
```

```json
{
  "mcpServers": {
    "awesome-copilot": {
      "command": "npx",
      "args": ["-y", "awesome-copilot-mcp", "start"]
    }
  }
}
```