# Homework 1: Introduction to Machine Learning — Instructions

This homework uses the pinned 2026 car fuel-efficiency release. Source: [`cohorts/2026/homework/01-intro/homework.md`](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/01-intro/homework.md) in the course repo.

## Setup

Install Python, NumPy, Pandas, Matplotlib, and Seaborn — see the environment setup material in the [`01-intro`](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/01-intro) module folder.

## Dataset

Download the 2026 Car Fuel Efficiency dataset from [here](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv), or:

```bash
wget https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv
```

Load it with pandas. (Local copy: [`../datasets/car_fuel_efficiency_2026.csv`](../datasets/car_fuel_efficiency_2026.csv))

## Questions

**Q1. Pandas version** — What version of pandas did you install? (`pd.__version__`, open text)

**Q2. Records count** — How many records are in the dataset?
`5000` / `9000` / `10000` / `15000`

**Q3. Fuel types** — How many fuel types are represented in the dataset?
`1` / `2` / `3` / `4`

**Q4. Missing values** — How many columns in the dataset have missing values?
`0` / `1` / `2` / `3` / `4`

**Q5. Max fuel efficiency** — What's the maximum fuel efficiency of cars from Asia?
`21.2` / `31.2` / `41.2` / `51.2`

**Q6. Median value of horsepower**
1. Find the median of `horsepower`.
2. Find the most frequent value (mode) of `horsepower`.
3. Fill missing values in `horsepower` with that mode.
4. Recompute the median.

Did it change? `Yes, it increased` / `Yes, it decreased` / `No`

**Q7. Sum of weights** (this is a mini linear regression, via the normal equation)
1. Select cars from Asia only.
2. Keep only `vehicle_weight` and `model_year`.
3. Take the first 7 rows.
4. Convert to a NumPy array `X`.
5. Compute `XTX = X.T @ X`.
6. Invert it: `XTX_inv`.
7. Let `y = [1100, 1300, 800, 900, 1000, 1100, 1200]`.
8. Compute `w = XTX_inv @ X.T @ y`.
9. What's the sum of the elements of `w`?

`0.0369` / `0.369` / `3.69` / `36.9`

## Submission

Submit at [courses.datatalks.club/ml-zoomcamp-2026/homework/hw01](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw01). The numeric options are calculated from the pinned 2026 release, so use whatever value your own calculation gives you. Optional bonus points for learning-in-public posts tagged `#mlzoomcamp`.