# Homework 2: Machine Learning for Regression

- Status: Finished
- Instructions: [`cohorts/2026/homework/02-regression/homework.md`](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/02-regression) in the course repo

## Objective

Build a linear regression model from scratch (via the normal equation) to predict `fuel_efficiency_mpg`, and compare how missing-value handling, regularization, and the train/val/test split seed each affect RMSE.

## Dataset

[`car_fuel_efficiency_2026.csv`](../datasets/car_fuel_efficiency_2026.csv), same pinned 2026 release used in Homework 1, already checksum-verified; see [`datasets/README.md`](../datasets/README.md). Only 5 columns are used here: `engine_displacement`, `horsepower`, `vehicle_weight`, `model_year`, `fuel_efficiency_mpg`.

## Approach

Implemented `train_linear_regression` and its regularized variant directly with NumPy `(XᵀX + rI)⁻¹Xᵀy`, plus an RMSE helper. Reused the exact shuffle-and-split logic given in the homework (60/20/20, `np.random.seed`). Compared 0-fill vs mean-fill for the one column with missing values, swept regularization strength `r`, checked split-seed sensitivity across 10 seeds, then trained a final regularized model on train+val and scored it on the held-out test set.

## Outcome

| Q | Question | Answer |
|---|----------|--------|
| 1 | Column with missing values | `horsepower` |
| 2 | Median horsepower | 254 |
| 3 | Better fill for missing values | With mean (RMSE 2.202 vs 2.205 for 0-fill) |
| 4 | Best regularization `r` | 0 |
| 5 | Std of RMSE across 10 seeds | 0.029 |
| 6 | Test RMSE (seed 9, r=0.001, train+val combined) | 2.236 |

Note on the EDA step: `fuel_efficiency_mpg` has a skew of about 0.08, essentially symmetric, no long tail. That's why no log transform is used anywhere in this homework, unlike some other modules' regression targets.