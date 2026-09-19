# Homework 4: Evaluation Metrics for Classification

- Status: Finished
- Instructions: [`cohorts/2026/homework/04-evaluation/homework.md`](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/04-evaluation)

## Objective

This homework was about evaluating classification quality beyond raw accuracy. I examined AUC by feature, swept thresholds for precision and recall, and used cross-validation to choose a regularization strength that generalized better than a single split.

## Dataset

The dataset is [`course_lead_scoring_2026.csv`](../datasets/course_lead_scoring_2026.csv), which is the same pinned course release used in Homework 3. I followed the assignment split and evaluation logic for the 2026 version.

## Approach

I used the same missing-value handling as the earlier classification homework, fit the logistic regression models specified in the assignment, and evaluated the system using AUC, precision, recall, F1, and 5-fold cross-validation. The threshold sweeps and the model-selection step were done directly from the homework workflow.

## Verified results

These are the values I recorded from the actual notebook work:

- Q1 strongest numerical feature by AUC: `lead_score`
- Q2 validation AUC: 0.732
- Q3 threshold where precision and recall are closest: 0.63
- Q4 threshold with the highest F1: 0.41
- Q5 standard deviation of AUC across 5 folds: 0.007
- Q6 best `C` from cross-validation: 0.001

## Outcome

| Q | Question | Answer |
|---|----------|--------|
| 1 | Numerical feature with highest AUC | `lead_score` |
| 2 | Model AUC on validation | 0.732 |
| 3 | Precision and recall intersection threshold | 0.63 |
| 4 | F1-maximizing threshold | 0.41 |
| 5 | Std of AUC across 5 folds | 0.007 |
| 6 | Best `C` value from cross-validation | 0.001 |