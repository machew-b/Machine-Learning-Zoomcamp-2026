# Homework 1: Introduction to Machine Learning

**Status:** Finished

## Objective

Build confidence with the course environment and basic pandas workflows by exploring a car fuel-efficiency dataset: checking library versions, counting records, inspecting categorical and numeric columns, and implementing a small linear regression by hand using the normal equation.

## Dataset

[`car_fuel_efficiency_2026.csv`](../datasets/car_fuel_efficiency_2026.csv) is the pinned 2026 release (10,000 rows × 11 columns). It was checksum-verified against the official course repo; see [`datasets/README.md`](../datasets/README.md).

## Approach

I loaded the dataset with pandas and answered each question using targeted operations: `.shape`, `.nunique()`, `.isnull().sum()`, `.median()`/`.mode()`/`.fillna()`, and for Q7, a manual normal-equation implementation with NumPy (`(XᵀX)⁻¹Xᵀy`).

## Outcome

| Q | Question | Answer |
|---|----------|--------|
| 1 | Pandas version | 3.0.5 |
| 2 | Records count | 10,000 |
| 3 | Fuel types | 3 (Gasoline, Diesel, Hybrid) |
| 4 | Columns with missing values | 2 (`horsepower`, `acceleration`) |
| 5 | Max fuel efficiency, Asia | 41.2 |
| 6 | Horsepower median after imputation | Decreased (254.0 → 252.0) |
| 7 | Sum of weights (normal equation) | ≈ 0.369 |