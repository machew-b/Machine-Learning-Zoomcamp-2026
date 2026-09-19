# Homework 3: Machine Learning for Classification

- Status: Finished
- Instructions: [`cohorts/2026/homework/03-classification/homework.md`](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/03-classification) in the course repo

## Objective

Build a logistic regression classifier predicting `converted` (whether a lead signed up), and explore feature relevance through correlation, mutual information, and feature elimination.

## Dataset

[`course_lead_scoring_2026.csv`](../datasets/course_lead_scoring_2026.csv), the pinned 2026 release, already checksum-verified; see [`datasets/README.md`](../datasets/README.md). 4 categorical features (`lead_source`, `industry`, `employment_status`, `location`), 4 numerical features (`annual_income`, `number_of_courses_viewed`, `interaction_count`, `lead_score`), target `converted`.

## Approach

Filled missing categorical values with `'NA'` and missing numerical values with `0.0`. Split 60/20/20 with the exact `train_test_split` calls given (`random_state=42` twice). Used `DictVectorizer` for one-hot encoding and `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)` as specified. For Q5, retrained the model with each candidate feature dropped one at a time and compared validation accuracy to the full-feature baseline.

## Outcome

| Q | Question | Answer |
|---|----------|--------|
| 1 | Mode of `industry` | `technology` |
| 2 | Most correlated numerical pair | `interaction_count` & `lead_score` (r ≈ 0.92) |
| 3 | Highest mutual information with `converted` | `lead_source` (0.03) |
| 4 | Validation accuracy (Q4 model) | 0.65 |
| 5 | Smallest accuracy difference on removal | `number_of_courses_viewed` (diff ≈ 0.002) |
| 6 | Best `C` for regularized model | 0.001 |