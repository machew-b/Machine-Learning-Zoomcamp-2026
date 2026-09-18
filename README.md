# Machine Learning Zoomcamp 2026

My homework and project submissions for [DataTalksClub's Machine Learning Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp) — a free, hands-on course covering the full ML engineering pipeline, from regression and classification through deployment, trees, deep learning, and model serving.

## Progress

| Homework | Module | Status | Date Submitted |
|----------|--------|--------|-----------------|
| [Homework 1](homework-01) | Introduction to Machine Learning | Pending | |
| [Homework 2](homework-02) | Machine Learning for Regression | Pending | |
| [Homework 3](homework-03) | Machine Learning for Classification | Pending | |
| [Homework 4](homework-04) | Evaluation Metrics for Classification | Pending | |
| [Homework 5](homework-05) | Deploying Machine Learning Models | Pending | |
| [Homework 6](homework-06) | Decision Trees and Ensemble Learning | Pending | |
| [Midterm Project](projects/midterm) | End-to-end project (Modules 1–6) | Pending | |
| [Homework 8](homework-08) | Neural Networks and Deep Learning | Pending | |
| [Homework 9](homework-09) | Serverless Deep Learning | Pending | |
| [Homework 10](homework-10) | Kubernetes and TensorFlow Serving | Pending | |
| [Capstone 1](projects/capstone-1) | End-to-end project (Modules 1–10) | Pending | |
| [Capstone 2](projects/capstone-2) | Optional second capstone | Pending | |

**Status key:** `Finished` — task complete · `In Progress` — currently working on it · `Pending` — not yet started.

## Repository Structure

Each homework folder follows the same layout:

```
homework-0X/
├── README.md            # Objective, approach, and outcome for this homework
├── INSTRUCTIONS.md       # The homework questions and dataset, as published for the 2026 cohort
└── homework-0X.ipynb    # My worked solutions
```

Project folders (midterm and capstones) follow a similar but slightly larger layout, since they include their own data prep, model, and (eventually) deployment code:

```
projects/<project-name>/
├── README.md            # Problem statement, dataset, approach, and results
├── notebook.ipynb        # EDA, model training, and evaluation
├── train.py              # Script to train and save the final model
├── predict.py             # Script/service to serve predictions
└── Dockerfile              # For containerized deployment
```

Shared datasets used across multiple homeworks live in the top-level [`datasets/`](datasets) folder and are referenced by each notebook using a relative path such as `../datasets/<file>.csv`.

## Course Info

- Course repo: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- Course platform (deadlines, submissions): [courses.datatalks.club/ml-zoomcamp-2026](https://courses.datatalks.club/ml-zoomcamp-2026/)
- Cohort: 2026 (started September 14, 2026)