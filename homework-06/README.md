# Homework 6: Decision Trees and Ensemble Learning

- Status: Finished
- Instructions: [`cohorts/2026/homework/06-trees/homework.md`](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/06-trees/homework.md)

## Objective

This homework focuses on tree-based regression for the fuel efficiency dataset. I trained a single decision tree, tuned a random forest across `n_estimators` and `max_depth`, inspected feature importance, and compared an XGBoost model against the reference setting.

## Dataset

The data comes from [`car_fuel_efficiency_2026.csv`](../datasets/car_fuel_efficiency_2026.csv), which is the same dataset version used in the course materials. I checked the dataset and the model workflow against the repository references before running the assignment.

## Approach

I filled missing values with zero, split the data with the required 60/20/20 train validation test pattern, and encoded the remaining columns with `DictVectorizer(sparse=True)`. I used the exact tree and forest configurations from the homework and checked the root split, validation RMSEs, and feature importances directly from the trained models. For the XGBoost comparison, I trained the model with the same setup and changed only `eta` as specified.

## Verified results

These are the values I obtained while working through the homework in the notebook:

- Q1 root split feature: `model_year`
- Q2 random forest RMSE, `n_estimators=10`: 1.837
- Q3 best `n_estimators`: 100, with RMSE 1.768
- Q4 best `max_depth`: 10, with mean RMSE 1.755
- Q5 most important feature among the listed candidates: `vehicle_weight`
- Q6 better `eta` setting for XGBoost: 0.1

## Outcome

| Q | Question | Answer |
|---|----------|--------|
| 1 | Feature used at the root split (`max_depth=1`) | `model_year` |
| 2 | RMSE for the random forest with `n_estimators=10` | 1.837 |
| 3 | Best `n_estimators` | 100 |
| 4 | Best `max_depth` | 10 |
| 5 | Most important feature among the listed candidates | `vehicle_weight` |
| 6 | Better `eta` for XGBoost | 0.1 |