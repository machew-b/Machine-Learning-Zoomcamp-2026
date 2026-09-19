# Homework 2: Machine Learning for Regression

- Status: Finished
- Instructions: [`cohorts/2026/homework/02-regression/homework.md`](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/02-regression)

## Objective

This assignment was about building and evaluating a linear regression model for `fuel_efficiency_mpg`. I compared missing-value handling, regularization, and split-seed sensitivity using the exact workflow from the homework.

## Dataset

The data comes from [`car_fuel_efficiency_2026.csv`](../datasets/car_fuel_efficiency_2026.csv), which is the same pinned 2026 release used in the course materials. I used the same feature subset and split logic specified in the assignment.

## Approach

I implemented the training logic directly with NumPy, checked the effect of different missing-value fills, swept the regularization strength, compared RMSE across random seeds, and then evaluated the final regularized model on the held-out test set.

## Verified results

These are the values I recorded while working through the homework in the notebook:

- Q1 missing-value column: `horsepower`
- Q2 median horsepower: 254
- Q3 better fill strategy: mean imputation
- Q4 best regularization value: 0
- Q5 standard deviation of RMSE across 10 seeds: 0.029
- Q6 final test RMSE: 2.236

## Outcome

| Q | Question | Answer |
|---|----------|--------|
| 1 | Column with missing values | `horsepower` |
| 2 | Median horsepower | 254 |
| 3 | Better fill for missing values | Mean fill |
| 4 | Best regularization `r` | 0 |
| 5 | Std of RMSE across 10 seeds | 0.029 |
| 6 | Test RMSE on the final model | 2.236 |