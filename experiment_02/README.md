# Experiment 02 — Per-Writer Authorship Classification

Build 19 independent binary classifiers (one per judge) trained on dissenting
opinions, then apply to full decision texts to estimate authorship probability.

## Setup

```bash
cd experiment_02
poetry install
```

Requires the UDPipe Czech model at `models/czech-pdt-ud-2.5-191206.udpipe`
(repo root). Download from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131

## Quick Start

```bash
# Full pipeline (all 7 steps)
poetry run python scripts/run_pipeline.py

# Use both classifiers (logistic + xgboost)
poetry run python scripts/run_pipeline.py --classifier both

# Resume from evaluation (skip UDPipe reprocessing)
poetry run python scripts/run_pipeline.py --from-step evaluate

# Select specific feature sets
poetry run python scripts/run_pipeline.py --features function_words surface char_ngrams pos_ngrams morphology
```

## Pipeline Steps

| Step | Description | Output |
|------|-------------|--------|
| 1. `load` | Load & filter CSV | `outputs/corpus.pkl` |
| 2. `udpipe_dissents` | Tokenize/tag dissents | `outputs/dissent_documents.pkl` |
| 3. `features_dissents` | Extract feature matrix | `outputs/dissent_features.pkl` |
| 4. `evaluate` | LOO-CV evaluation | `outputs/loo_results.pkl`, `outputs/results_summary.txt` |
| 5. `udpipe_decisions` | Tokenize/tag decisions | `outputs/decision_documents.pkl` |
| 6. `features_decisions` | Extract features | `outputs/decision_features.pkl` |
| 7. `apply` | Score decisions | `outputs/authorship_probabilities_*.csv` |

## CLI Options

```
--min-dissents N     Minimum dissents per author (default: 5)
--model-path PATH    Path to UDPipe model
--features SET [..]  Feature sets: function_words surface char_ngrams pos_ngrams morphology
--classifier TYPE    logistic | xgboost | both (default: logistic)
--from-step STEP     Resume from step (default: load)
```

## Project Structure

```
experiment_02/
├── IMPLEMENTATION.md          # Detailed design document
├── README.md                  # This file
├── pyproject.toml
├── outputs/                   # Auto-generated
├── scripts/
│   └── run_pipeline.py
└── src/
    └── fingerprint/
        ├── __init__.py
        ├── data_loader.py
        ├── preprocessing.py
        ├── classifiers.py
        ├── evaluation.py
        ├── feature_importance.py
        └── features/
            ├── __init__.py
            ├── function_words.py
            ├── ngrams.py
            ├── surface.py
            └── morphology.py
```

## See Also

- [IMPLEMENTATION.md](IMPLEMENTATION.md) — Full design document with rationale
- [experiment_01/](../experiment_01/) — Multi-class baseline (Experiment 01)
