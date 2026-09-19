# Machine Learning Zoomcamp 2026

This repository contains my work for [DataTalksClub's Machine Learning Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp), a hands-on course covering the full ML engineering workflow from regression and classification to deployment, trees, deep learning, and model serving.

## Progress

| Homework | Module | Status | Date Submitted |
|----------|--------|--------|-----------------|
| [Homework 1](homework-01) | Introduction to Machine Learning | Finished | |
| [Homework 2](homework-02) | Machine Learning for Regression | Pending | |
| [Homework 3](homework-03) | Machine Learning for Classification | Pending | |
| [Homework 4](homework-04) | Evaluation Metrics for Classification | Pending | |
| [Homework 5](homework-05) | Deploying Machine Learning Models | Pending | |
| [Homework 6](homework-06) | Decision Trees and Ensemble Learning | Pending | |
| [Midterm Project](projects/midterm) | End-to-end project (Modules 1-6) | Pending | |
| [Homework 8](homework-08) | Neural Networks and Deep Learning | Pending | |
| [Homework 9](homework-09) | Serverless Deep Learning | Pending | |
| [Homework 10](homework-10) | Kubernetes and TensorFlow Serving | Pending | |
| [Capstone 1](projects/capstone-1) | End-to-end project (Modules 1-10) | Pending | |
| [Capstone 2](projects/capstone-2) | Optional second capstone | Pending | |

**Status key:** `Finished` — task complete · `In Progress` — currently working on it · `Pending` — not yet started.

## Repository structure

Each homework folder follows the same basic layout:

```text
homework-0X/
├── homework-0X.ipynb    # My worked solutions
├── README.md            # Objective, source link, approach, and outcome for the homework
└── .ipynb_checkpoints/
```

Project folders follow a similar but slightly larger structure since they include their own data preparation, model training, and deployment code:

```text
projects/<project-name>/
├── Dockerfile           # For containerized deployment
├── notebook.ipynb       # EDA, model training, and evaluation
├── predict.py           # Model-serving script or API
├── README.md            # Problem statement, dataset, approach, and results
├── train.py             # Script to train and save the final model
└── requirements.txt     # Optional project dependencies
```

Shared datasets used across multiple homeworks live in the top-level [`datasets/`](datasets) folder and are referenced by each notebook using paths such as `../datasets/<file>.csv`.

## Quick start

See [`SETUP.md`](SETUP.md) for a reproducible environment setup using a Python virtual environment and pinned dependencies.

## Repository conventions

- Keep notebooks, generated artifacts, and local caches out of version control via [`.gitignore`](.gitignore).
- Document each assignment with a brief objective, dataset, approach, and result in its own `README.md`.
- Use the pinned versions in [`requirements.txt`](requirements.txt) when possible for consistent results.
- Keep relative paths stable by running notebooks from the repository root.

## Course info

- Course repo: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- Course platform (deadlines, submissions): [courses.datatalks.club/ml-zoomcamp-2026](https://courses.datatalks.club/ml-zoomcamp-2026/)
- Cohort: 2026 (started September 14, 2026)