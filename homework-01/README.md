# Homework 1: Introduction to Machine Learning

- Status: Finished
- Instructions: [`cohorts/2026/homework/01-intro/homework.md`](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/01-intro/homework.md)

## Objective

This homework was about getting comfortable with the course workflow and the dataset. I inspected the fuel-efficiency data, checked the basic pandas operations, and implemented a small linear regression by hand using the normal equation.

## Dataset

The data comes from [`car_fuel_efficiency_2026.csv`](../datasets/car_fuel_efficiency_2026.csv), which is the pinned 2026 release. I checked the dataset and the workbook against the repository references before using it.

## Approach

I loaded the dataset with pandas and worked through the required checks: record counts, categorical coverage, missing-value counts, imputation behavior, and the hand-written normal-equation solution for the final regression step.

## Verified results

These are the values I recorded while working through the actual homework in the notebook:

- Q1 pandas version: 3.0.5
- Q2 number of records: 10,000
- Q3 fuel types: 3
- Q4 columns with missing values: `horsepower` and `acceleration`
- Q5 maximum fuel efficiency in Asia: 41.2
- Q6 horsepower median after imputation: 252.0
- Q7 sum of weights from the normal equation: approximately 0.369

## Outcome

| Q | Question | Answer |
|---|----------|--------|
| 1 | Pandas version | 3.0.5 |
| 2 | Records count | 10,000 |
| 3 | Fuel types | 3 |
| 4 | Columns with missing values | `horsepower`, `acceleration` |
| 5 | Max fuel efficiency in Asia | 41.2 |
| 6 | Horsepower median after imputation | 252.0 |
| 7 | Sum of weights from the normal equation | 0.369 |