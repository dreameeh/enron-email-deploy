# Enron Email Processing Pipeline

This repository contains an optimized pipeline for processing and analyzing the Enron email dataset on resource-constrained systems like a Mac Mini.

## Setup

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Place your raw Enron email data in the `data/raw` directory.

## Usage

### 1. Data Processing

The pipeline is designed to work with limited memory by:
- Processing data in batches
- Using memory-efficient data types
- Implementing early stopping in training
- Using smaller language models

Start with a sample dataset:
```bash
python src/data/process_emails.py
```
This will create a sample dataset in `data/samples/email_sample.parquet`.

### 2. Feature Engineering

Generate features from the processed data:
```bash
python src/features/feature_engineering.py
```
This will create features in `data/processed/email_features.parquet`.

## Memory Optimization Tips

1. Adjust batch sizes in `process_emails.py` based on your available memory
2. Modify `max_features` in `feature_engineering.py` to control TF-IDF vocabulary size
3. Use `pyarrow` parquet format for efficient data storage
4. Monitor memory usage and adjust parameters accordingly

## Project Structure

```
enron-email-deploy/
├── data/
│   ├── raw/          # Original email data
│   ├── processed/    # Cleaned and processed data
│   └── samples/      # Small data samples for testing
├── src/
│   ├── data/         # Data processing scripts
│   ├── features/     # Feature engineering
│   └── models/       # Model training (to be implemented)
├── notebooks/        # Experimental notebooks
└── requirements.txt  # Dependencies
```
