# Technical Summary — First ML Pipeline (Week 06 · Day 3)

*Plain-language write-up for a non-technical reader.*

## The dataset

We examined 600 students. For each one we had: reported study hours per week,
reported sleep hours per night, attendance percentage, which class section they
were in (A, B, or C), and their exam score. The dataset was created with a fixed
random seed, so it is fully reproducible and we can check every result against
the known true relationships.

## What we built

We created two prediction models:

1. **A regression model** that predicts the actual exam score (a number)
2. **A classification model** that predicts whether a student will achieve
   distinction (score ≥ 85)

Both models were compared against "dumb baselines" — the simplest possible
predictors that don't learn from the features at all.

## Key findings

### 1. Predicting exam scores

**The baseline** always predicted the average exam score (~88 points) for every
student, regardless of their study habits, sleep, or attendance. This resulted
in typical errors of about 10 points.

**Our linear regression model** reduced those errors to about 7 points — a 31%
improvement. It explains about 53% of the variation in exam scores, meaning
study hours, sleep, attendance, and class section together account for roughly
half of why students score differently.

**The strongest predictor** was study hours per week — each additional hour was
associated with about 2 more points on the exam. Section C students scored
about 3 points higher than Section A students (matching the dataset's design).

### 2. Predicting distinction students

**The baseline** always predicted "distinction" for every student, since 65.3%
of students scored ≥ 85. This achieved 65% accuracy but was completely useless
for its intended purpose — it couldn't distinguish between students who would
and wouldn't achieve distinction.

**Our logistic regression model** achieved 77% accuracy while actually
identifying distinction students: 83% recall (caught 83% of actual distinction
students) and 81% precision (81% of predicted distinctions were correct). This
demonstrates why accuracy alone can be misleading on imbalanced data.

### 3. Why accuracy alone is misleading

The baseline achieved 65% accuracy by always predicting the majority class
(distinction). This looks reasonable at first glance, but the model has learned
nothing — it can't identify a single student who won't achieve distinction. The
recall of 100% and precision of 65% reveal this: it catches everyone but
produces many false alarms. The real model's 77% accuracy with 83% recall and
81% precision is genuinely useful.

## What checks we ran before calling it done

- **Train/test split**: We split the data once (80/20) with a fixed seed and
  used the same split for every model. This ensures fair comparison.
- **Baseline comparison**: Every real model must beat its respective baseline
  to be worth using.
- **Self-audit**: Every number in the report was independently recomputed and
  verified against a fresh calculation.
- **Automated verification**: A 64-check test suite validates the dataset,
  feature engineering, split, models, and metrics.

## Honest limitations

- **Correlation ≠ causation**: We cannot say studying more *causes* higher
  scores — only that they're correlated. A motivated student might both study
  more and have better preparation habits not captured in the data.
- **Synthetic data**: The model was trained on 600 synthetic students. Real
  students might have different patterns.
- **Limited features**: We only have 5 features. Many real factors (teaching
  quality, prior knowledge, exam difficulty) are not included.
- **Linear assumptions**: Both models assume linear relationships. Real
  relationships might be curved or interact in complex ways.
- **Single split**: We used one fixed train/test split. Cross-validation would
  give more robust estimates.
- **Threshold sensitivity**: The distinction threshold (85) was chosen
  arbitrarily. Different thresholds would change the class balance and metrics.

## What would improve this

- Cross-validation instead of a single split
- More features (prior GPA, assignment completion, etc.)
- Non-linear models (random forest, gradient boosting)
- Hyperparameter tuning
- Calibration analysis for the classification model
