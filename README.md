# First ML Pipeline — Baselines, Linear & Logistic Regression, and Metrics That Don't Lie

## Overview

This notebook builds a complete, reproducible machine learning pipeline that demonstrates the fundamental principles of model development: honest baselines, proper train/test splitting, and meaningful metrics.

## Learning Objectives

1. **Train/Test Split**: Why a model must never be evaluated on the data it was trained on
2. **Baselines**: Building the "dumbest possible baseline" to establish a performance floor
3. **Feature Engineering**: Converting categorical variables to numeric form using one-hot encoding
4. **Linear Regression**: Predicting continuous values with RMSE and R² metrics
5. **Logistic Regression**: Predicting binary outcomes with accuracy, precision, recall, and F1
6. **Reproducibility**: Assembling the whole flow as one fixed-seed, reproducible pipeline

## Key Findings

### Regression (Predicting Exam Scores)
- Linear regression reduces prediction error by ~40-50% compared to the baseline
- Study hours per week is the strongest predictor (~2.5 points per hour)
- Model explains ~40-50% of variance in exam scores

### Classification (Predicting Distinction Students)
- Baseline achieves ~70% accuracy by always predicting "no distinction" — but catches 0% of actual distinction students
- Logistic regression catches ~70% of distinction students with ~85% precision
- Demonstrates why accuracy alone is misleading on imbalanced data

## Dataset

Synthetic student performance data (n=600) with known relationships:
- `study_hours_per_week`: Weekly study hours (normal distribution, mean=10)
- `sleep_hours_per_night`: Nightly sleep hours (normal distribution, mean=7)
- `attendance_pct`: Class attendance percentage (normal distribution, mean=85)
- `class_section`: Class section (A, B, or C)
- `exam_score`: Final exam score (0-100, derived from other features)

## Files

- `ml_pipeline.ipynb` — Main notebook with complete ML pipeline
- `ml_pipeline_executed.ipynb` — Executed notebook with all outputs
- `requirements.txt` — Python dependencies
- `README.md` — This file

## How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Open `ml_pipeline.ipynb` in Jupyter
3. Run all cells (Kernel → Restart & Run All)

## Reproducibility

All random operations are seeded:
- Dataset generation: `seed=21`
- Train/test split: `random_state=42`
- Logistic regression: `random_state=42`

Two runs will produce identical results.

## Metrics Explained

### Regression Metrics
- **RMSE (Root Mean Squared Error)**: Average prediction error in the same units as the target
- **R² (Coefficient of Determination)**: Proportion of variance explained (0 = baseline, 1 = perfect)

### Classification Metrics
- **Accuracy**: Overall fraction of correct predictions
- **Precision**: Of predicted positives, how many are actually positive
- **Recall**: Of actual positives, how many were caught
- **F1**: Harmonic mean of precision and recall

## Limitations

- Causation cannot be claimed from correlation
- Model trained on synthetic data may not generalize to real students
- Only 5 features used — many real factors omitted
- Linear assumptions may not hold for all relationships
- Single train/test split used (cross-validation would be more robust)
