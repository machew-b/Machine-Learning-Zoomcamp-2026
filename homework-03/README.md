# Homework 3: Machine Learning for Classification

- Status: Finished
- Instructions: [`cohorts/2026/homework/03-classification/homework.md`](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/03-classification)

## Objective

This homework was about building a logistic regression model for lead conversion and checking which features matter most. I used the course dataset, the required train validation test split, and the exact model setup from the assignment.

## Dataset

The dataset is [`course_lead_scoring_2026.csv`](../datasets/course_lead_scoring_2026.csv). I used the same release as the course materials and checked the relevant workflow before fitting the model.

## Approach

I filled missing categorical values with `NA`, filled missing numerical values with zero, encoded the categorical features with `DictVectorizer`, and trained the specified logistic regression model. I then checked correlations, mutual information, and feature removal effects to answer the homework questions.

## Verified results

These are the values I recorded while working through the actual assignment:

- Q1 mode of `industry`: `technology`
- Q2 most correlated numerical pair: `interaction_count` and `lead_score`
- Q3 highest mutual information with `converted`: `lead_source`
- Q4 validation accuracy: 0.65
- Q5 smallest accuracy drop after feature removal: `number_of_courses_viewed`
- Q6 best `C` value: 0.001

## Outcome

| Q | Question | Answer |
|---|----------|--------|
| 1 | Mode of `industry` | `technology` |
| 2 | Most correlated numerical pair | `interaction_count` and `lead_score` |
| 3 | Highest mutual information with `converted` | `lead_source` |
| 4 | Validation accuracy | 0.65 |
| 5 | Smallest accuracy change after removing one feature | `number_of_courses_viewed` |
| 6 | Best `C` for the regularized model | 0.001 |