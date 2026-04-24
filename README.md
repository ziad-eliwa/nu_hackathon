# Arabic ABSA (Aspect-Based Sentiment Analysis)

An Arabic Aspect-Based Sentiment Analysis pipeline for restaurant reviews. Detects aspects (food, service, price, etc.) and predicts sentiment (positive/negative/neutral) for each aspect.

## Features

- **Multi-label aspect detection** using ensemble of TF-IDF + Transformer models
- **Aspect-conditioned sentiment classification**
- **Semi-supervised learning** with pseudo-labeling on unlabeled data
- **Calibrated thresholds** per aspect for optimal performance
- **Schema validation** to ensure valid output format

## Quick Start

```bash
# Install dependencies
uv sync

# Train Path A (aspect detection)
python main.py train-aspect

# Train Path B (sentiment classification)
python main.py train-sentiment

# Run inference
python main.py predict --input-csv data/DeepX_validation.csv --output-json predictions.json

# Evaluate predictions
python main.py evaluate

# Semi-supervised training with pseudo-labeling
python main.py semi-supervised

# Launch interactive GUI
uv run streamlit run src/absa/gui/app.py
```

## GUI Workflow

The Streamlit app supports:

- Uploading a CSV test set.
- Running integrated Path A (aspects) + Path B (sentiment) inference.
- Visualizing aspect frequencies and sentiment distributions.
- Viewing per-review predictions in a table.
- Downloading results as CSV and submission JSON.

Required CSV columns:

- `review_text`

Recommended columns:

- `review_id`, `star_rating`, `date`, `business_name`, `business_category`, `platform`

If `review_id` is missing or empty, the GUI auto-generates IDs.

## Architecture

### Aspect Taxonomy
- food, service, price, cleanliness, delivery, ambiance, app_experience, general, none

### Sentiment Labels
- positive, negative, neutral

### Hard Rules
- If `none` is predicted, sentiment must be `neutral`
- `none` cannot appear with other aspects

## Commands

| Command | Description |
|---------|-------------|
| `train-aspect` | Train aspect detection models (TF-IDF + Transformer ensemble) |
| `train-sentiment` | Train aspect-conditioned sentiment classifier |
| `predict` | Run end-to-end inference on input CSV |
| `evaluate` | Evaluate predictions against validation set |
| `semi-supervised` | Train with pseudo-labeling on unlabeled data |

### Predict Options
```bash
python main.py predict --input-csv data/DeepX_validation.csv --output-json predictions.json --model-dir artifacts
```

### Semi-supervised Options
```bash
python main.py semi-supervised --unlabeled-csv data/DeepX_unlabeled.csv --confidence-threshold 0.85
```

## Project Structure

```
src/absa/
├── config/          # Taxonomy, settings, constants
├── data/            # IO, schemas, validation
├── preprocess/      # Text normalization, metadata extraction
├── features/        # TF-IDF feature extraction
├── models/          # Aspect & sentiment models
├── training/        # Training & calibration scripts
├── inference/       # Prediction & submission packaging
└── evaluation/      # Metrics & error analysis
```

## Data

- `data/DeepX_train.csv` - Labeled training data (1,971 samples)
- `data/DeepX_validation.csv` - Validation data (500 samples)
- `data/DeepX_unlabeled.csv` - Unlabeled data (7,047 samples)

## Results

### After Semi-supervised Training
- **Aspect micro_f1**: 0.786
- **Aspect macro_f1**: 0.719
- **Sentiment macro_f1**: 0.671
- **Tuple F1**: 0.688
- **Coverage**: 78.5% of gold aspects predicted

### Per-Aspect F1
| Aspect | F1 |
|--------|-----|
| service | 0.856 |
| app_experience | 0.879 |
| food | 0.778 |
| general | 0.727 |
| delivery | 0.758 |
| ambiance | 0.755 |
| price | 0.705 |
| cleanliness | 0.563 |
| none | 0.444 |