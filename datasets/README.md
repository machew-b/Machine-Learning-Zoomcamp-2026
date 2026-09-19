# Datasets

This folder stores the pinned dataset snapshots used throughout the 2026 Machine Learning Zoomcamp cohort. Keeping a local copy ensures the notebooks remain reproducible and reduces surprises caused by upstream data changes or environment differences.

For the official dataset documentation, generation process, and quality notes, see the course repo's [release notes](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/data/README.md).

| File | Description | Rows | Used in | SHA-256 |
|------|-------------|------|---------|---------|
| [`car_fuel_efficiency_2026.csv`](car_fuel_efficiency_2026.csv) | Vehicle attributes and fuel-economy metrics for regression and feature analysis | 10,000 | Homework 1, Homework 2 | `00a5cab178a8b7cd6e9157ee71dabc236ba7f20b1832336d358b07af14ba431f` |
| [`course_lead_scoring_2026.csv`](course_lead_scoring_2026.csv) | Lead-level characteristics used for classification and conversion prediction | 5,000 | Homework 3, Homework 4 | `2d9da196bdefd2a45aa0a17ca21c19f06a5d9d56af628be1f04bbb3531cc12e9` |

## Using these files

These CSVs are intended to be read from the repository using relative paths such as:

```python
import pandas as pd

df = pd.read_csv("../datasets/car_fuel_efficiency_2026.csv")
```

## Verifying a file

To confirm the downloaded file matches the signed dataset snapshot, compute its SHA-256 hash and compare it to the value in the table above.

```bash
sha256sum car_fuel_efficiency_2026.csv
```

Windows PowerShell:

```powershell
Get-FileHash car_fuel_efficiency_2026.csv -Algorithm SHA256
```

## Reference environment

The homework answer keys were generated with Python 3.11.15, NumPy 2.3.3, Pandas 2.3.2, Scikit-Learn 1.7.2, and XGBoost 3.2.0 (for tree-based homeworks). Matching these versions locally helps avoid small library-version differences that can shift a computed result into a different multiple-choice bucket.