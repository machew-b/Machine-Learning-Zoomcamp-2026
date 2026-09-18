# Datasets

Local copies of the official ML Zoomcamp 2026 datasets, used across multiple homeworks. Sourced from the course repo's [`cohorts/2026/data`](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/data/README.md) release — see that page for how the course team generates and validates them; this file tracks what's in *my* copy and confirms it's unmodified.

## Files

| File | Rows | Used in | SHA-256 | Verified |
|------|------|---------|---------|----------|
| [`car_fuel_efficiency_2026.csv`](car_fuel_efficiency_2026.csv) | 10,000 | Homework 1, Homework 2 | `00a5cab178a8b7cd6e9157ee71dabc236ba7f20b1832336d358b07af14ba431f` | ✅ matches official release |
| [`course_lead_scoring_2026.csv`](course_lead_scoring_2026.csv) | 5,000 | Homework 3, Homework 4 | `2d9da196bdefd2a45aa0a17ca21c19f06a5d9d56af628be1f04bbb3531cc12e9` | ✅ matches official release |
| [`car_fuel_efficiency_2026_report.json`](car_fuel_efficiency_2026_report.json) | — | data-quality report for the car dataset | *(varies — timestamped)* | — |
| [`course_lead_scoring_2026_report.json`](course_lead_scoring_2026_report.json) | — | data-quality report for the lead dataset | *(varies — timestamped)* | — |
| [`validate_release.py`](validate_release.py) | — | official release-integrity check | — | ✅ ran locally, printed `ML Zoomcamp 2026 data release: OK` |
| [`validate_homework.py`](validate_homework.py) | — | course team's answer/option-consistency checker | — | reference only — see note below |

To re-verify a CSV matches the official release after re-downloading it:

```bash
sha256sum car_fuel_efficiency_2026.csv
# should print: 00a5cab178a8b7cd6e9157ee71dabc236ba7f20b1832336d358b07af14ba431f

sha256sum course_lead_scoring_2026.csv
# should print: 2d9da196bdefd2a45aa0a17ca21c19f06a5d9d56af628be1f04bbb3531cc12e9
```

Or run the full check, which also confirms bound violations, missingness rates, and correlation structure (not just file integrity):

```bash
python validate_release.py
# ML Zoomcamp 2026 data release: OK
```

**Note on `validate_homework.py`:** this is the course team's own tool for recomputing reference answers for HW1–HW4 and HW6 and confirming each one is listed among that homework's multiple-choice options. It expects the official repo's folder layout (`cohorts/2026/homework/<module>/homework.md`), so it won't run as-is against my repo's `homework-0X/` structure unless those paths are mirrored. I just kept here for reference/provenance, not meant to be run directly in this repo.

## Reference environment

Per the course's [release notes](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/data/README.md), the homework's numeric answer options were generated against:

- Python 3.11.15
- NumPy 2.3.3
- Pandas 2.3.2
- Scikit-Learn 1.7.2
- XGBoost 3.2.0 (for tree-based homeworks)

Matching these versions locally (or getting close) reduces the chance that a library-version quirk shifts a computed value into a different multiple-choice bucket than intended.

## Notes on the data

**[`car_fuel_efficiency_2026.csv`](car_fuel_efficiency_2026.csv)** (regression, [Homework 1](../homework-01), Homework 2): synthetic but physically-consistent car measurements, derived in dependency order so e.g. MPG stays in a plausible range (19.8–41.2) for the given specs, per [`car_fuel_efficiency_2026_report.json`](car_fuel_efficiency_2026_report.json). `horsepower` is missing in 8.8% of rows (more common for older vehicles); `acceleration` is missing in 2.6% of rows with a mild origin-dependent pattern. Strongest relationships: `engine_displacement`↔`horsepower` (r≈0.76) and `engine_displacement`↔`vehicle_weight` (r≈0.69).

**[`course_lead_scoring_2026.csv`](course_lead_scoring_2026.csv)** (classification, Homework 3, Homework 4): a marketing-funnel dataset with genuine feature correlations and a probabilistic (not hard-coded) conversion label — 57.7% converted overall, per [`course_lead_scoring_2026_report.json`](course_lead_scoring_2026_report.json). `interaction_count` is by far the strongest predictor of `lead_score` (r≈0.93), ahead of `number_of_courses_viewed` (r≈0.77). Missingness is conditional rather than independent: `annual_income` is missing in ~7.4% of rows overall, but at a meaningfully higher rate for students than for employed leads — worth remembering if a homework question asks about missingness patterns rather than just counts.

## Source

- Release notes: [`cohorts/2026/data/README.md`](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/data/README.md) in the course repo
- Course repo: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)