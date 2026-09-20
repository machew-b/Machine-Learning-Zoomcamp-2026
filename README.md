# Machine Learning Zoomcamp 2026

This repository contains my work for the 2026 cohort of [DataTalksClub's Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp). The project is organized by homework and follows the course sequence from the foundations of ML through model deployment and serving.

## Completed work

The table below tracks the homework folders that are present in this repository. In this tracker, the date is recorded as the official course deadline in [Course Management](https://courses.datatalks.club/ml-zoomcamp-2026/), which is the same date used for the homework submission window.

| Homework | Topic | Status | Date submitted |
|----------|-------|--------|----------------|
| [Homework 1](homework-01) | Introduction to Machine Learning | Completed | 28 Sep 2026 |
| [Homework 2](homework-02) | Machine Learning for Regression | Completed | 5 Oct 2026 |
| [Homework 3](homework-03) | Machine Learning for Classification | Completed | 12 Oct 2026 |
| [Homework 4](homework-04) | Evaluation Metrics for Classification | Completed | 19 Oct 2026 |
| [Homework 5](homework-05) | Deploying Machine Learning Models | Completed | 26 Oct 2026 |
| [Homework 6](homework-06) | Decision Trees and Ensemble Learning | Completed | 2 Nov 2026 |
| [Homework 8](homework-08) | Neural Networks and Deep Learning | Completed | 16 Nov 2026 |
| [Homework 9](homework-09) | Serverless Inference | Completed | 23 Nov 2026 |
| [Homework 10](homework-10) | Kubernetes and Model Serving | Completed | 30 Nov 2026 |

## Repository layout

Each homework folder contains its own writeup and the files needed to reproduce that assignment. The exact content varies by homework, but the usual structure is:

```text
homework-0X/
├── README.md
├── notebook or training script
├── supporting artifacts when needed
└── verification output or proof files when required
```

Shared datasets used across the course are stored in the top-level [`datasets/`](datasets) folder and are referenced by notebooks and scripts using relative paths.

## Setup

See [`SETUP.md`](SETUP.md) for the local environment setup and the pinned dependency workflow used across the course.

## Notes

- I keep notebooks when they are useful records of the work, but the course does not require every homework to use an `.ipynb` file. Scripts and checked results are appropriate for assignments such as deployment, serverless, and Kubernetes.
- I keep generated artifacts in the repository only when they are part of the assignment. Local environments, downloaded datasets, caches, and temporary run directories stay out of Git.
- Each homework folder includes a short explanation of the objective, the approach, and the result.
- The project is kept clean and easy to review so the work reads like the actual course work.

## Course links

- Course repo: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- Course platform: [Machine Learning Zoomcamp 2026](https://courses.datatalks.club/ml-zoomcamp-2026/)
- Course Management: [Course Management](https://courses.datatalks.club/ml-zoomcamp-2026/)
- Cohort: 2026