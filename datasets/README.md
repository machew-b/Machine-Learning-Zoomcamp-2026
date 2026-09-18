# Datasets

This folder contains the pinned ML Zoomcamp 2026 datasets used across the course assignments. They are the official release files from the course repository, stored locally so the project can work from a stable and reproducible copy.

For the original course documentation, see the [release notes](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/data/README.md). This README explains what is in the folder and how to verify it.

## Quick summary

| File | Rows | Used in | SHA-256 | Status |
|------|------|---------|---------|--------|
| [`car_fuel_efficiency_2026.csv`](car_fuel_efficiency_2026.csv) | 10,000 | Homework 1, Homework 2 | `00a5cab178a8b7cd6e9157ee71dabc236ba7f20b1832336d358b07af14ba431f` | ✅ official release match |
| [`course_lead_scoring_2026.csv`](course_lead_scoring_2026.csv) | 5,000 | Homework 3, Homework 4 | `2d9da196bdefd2a45aa0a17ca21c19f06a5d9d56af628be1f04bbb3531cc12e9` | ✅ official release match |
| [`car_fuel_efficiency_2026_report.json`](car_fuel_efficiency_2026_report.json) | — | data-quality report for the car dataset | *(timestamped; varies by run)* | — |
| [`course_lead_scoring_2026_report.json`](course_lead_scoring_2026_report.json) | — | data-quality report for the lead dataset | *(timestamped; varies by run)* | — |
| [`validate_release.py`](validate_release.py) | — | validation script for the dataset release | — | ✅ passed locally |
| [`validate_homework.py`](validate_homework.py) | — | course team's answer/option consistency checker | — | reference only |

## How to verify the CSVs

From this folder, run the following commands:

### macOS / Linux

```bash
sha256sum car_fuel_efficiency_2026.csv
# expected: 00a5cab178a8b7cd6e9157ee71dabc236ba7f20b1832336d358b07af14ba431f

sha256sum course_lead_scoring_2026.csv
# expected: 2d9da196bdefd2a45aa0a17ca21c19f06a5d9d56af628be1f04bbb3531cc12e9
```

### Windows PowerShell

```powershell
Get-FileHash .\car_fuel_efficiency_2026.csv -Algorithm SHA256
# expected: 00a5cab178a8b7cd6e9157ee71dabc236ba7f20b1832336d358b07af14ba431f

Get-FileHash .\course_lead_scoring_2026.csv -Algorithm SHA256
# expected: 2d9da196bdefd2a45aa0a17ca21c19f06a5d9d56af628be1f04bbb3531cc12e9
```

### Full data-quality check

```bash
python validate_release.py
# ML Zoomcamp 2026 data release: OK
```

This script checks that the data has the expected shapes, required target columns, reasonable missingness, and acceptable correlation structure. It does not compare SHA-256 hashes; use the checksum commands above for file identity.

**Note on `validate_homework.py`:** this is the course team's tool for recomputing reference answers for HW1–HW4 and HW6, then checking that each value appears in the corresponding multiple-choice options. It expects the official course repo layout (`cohorts/2026/homework/<module>/homework.md`), so it will not run as-is against this repository's `homework-0X/` structure unless those paths are mirrored. I kept it here only for reference and provenance.

## Reference environment

The course release notes say the homework numeric answer options were generated with:

- Python 3.11.15
- NumPy 2.3.3
- Pandas 2.3.2
- Scikit-Learn 1.7.2
- XGBoost 3.2.0 (for tree-based homeworks)

Using similar versions locally reduces the chance that a library-version difference causes a computed value to land in a different multiple-choice bucket.

## What the data looks like

### [`car_fuel_efficiency_2026.csv`](car_fuel_efficiency_2026.csv)

This is the regression dataset used in Homework 1 and Homework 2. It is synthetic but physically consistent: measurements were generated in dependency order so values like MPG remain in a plausible range (19.8–41.2) for the stated vehicle specs.

Key details:

- `horsepower` is missing in 8.8% of rows, and the missingness is more common in older vehicles.
- `acceleration` is missing in 2.6% of rows, with a mild origin-dependent pattern.
- Notable relationships: `engine_displacement` ↔ `horsepower` (r ≈ 0.76), `engine_displacement` ↔ `vehicle_weight` (r ≈ 0.69), and `model_year` ↔ `fuel_efficiency_mpg` (r ≈ 0.50).

### [`course_lead_scoring_2026.csv`](course_lead_scoring_2026.csv)

This is the classification dataset used in Homework 3 and Homework 4. It follows a funnel-like pattern in which engagement influences later behavior, and the conversion label is probabilistic rather than hard-coded.

Key details:

- 57.7% of leads converted overall.
- `interaction_count` has the strongest linear association with `lead_score` (r ≈ 0.93), ahead of `number_of_courses_viewed` (r ≈ 0.77).
- Missingness is conditional rather than independent: `annual_income` is missing in about 7.4% of rows overall, and the missing rate is meaningfully higher for students than for employed leads.

## Source

- Release notes: [`cohorts/2026/data/README.md`](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/data/README.md) in the course repo
- Course repo: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)