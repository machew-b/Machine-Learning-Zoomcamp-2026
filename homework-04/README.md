# Homework 4: Evaluation Metrics for Classification

- Status: Finished
- Instructions: [`cohorts/2026/homework/04-evaluation/homework.md`](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/04-evaluation) in the course repo

## Objective

Move past plain accuracy: rank numerical features by AUC, evaluate the Homework 3-style logistic regression with AUC/precision/recall/F1 across thresholds, and use K-fold cross-validation to pick a regularization strength that generalizes rather than just fitting one split well.

## Dataset

[`course_lead_scoring_2026.csv`](../datasets/course_lead_scoring_2026.csv), the pinned 2026 release, already checksum-verified; see [`datasets/README.md`](../datasets/README.md). Same features as Homework 3, but this homework uses `random_state=1` for splitting (Homework 3 used 42), so the actual train/val/test rows differ between the two.

## Approach

Same missing-value handling as Homework 3 (categorical to `'NA'`, numerical to `0.0`). For Q1, treated each numerical column as a raw prediction score and computed AUC directly against `converted`, flipping the sign for any column with AUC below 0.5. For Q2 onward, one-hot encoded with `DictVectorizer` and trained `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)`, no `random_state` this time, matching the homework's exact model call. Swept thresholds 0.00 to 1.00 in steps of 0.01 for precision, recall, and F1. For Q5 and Q6, used `KFold(n_splits=5, shuffle=True, random_state=1)` over `df_full_train`, refitting a fresh `DictVectorizer` inside each fold.

## Outcome

| Q | Question | Answer |
|---|----------|--------|
| 1 | Numerical feature with highest AUC | `lead_score` (AUC ≈ 0.79) |
| 2 | Model AUC on validation | 0.732 |
| 3 | Precision/recall intersection threshold | 0.63 |
| 4 | F1-maximizing threshold | 0.41 |
| 5 | Std of AUC across 5 folds (C=1.0) | 0.007 |
| 6 | Best `C` from cross-validation | 0.001 (mean AUC 0.749, beating both 1 and 1e-6 outright) |