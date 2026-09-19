# Machine Learning Zoomcamp 2026

This repository contains my work for [DataTalksClub's Machine Learning Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp). It is organized by homework and follows the course sequence from the early ML foundations through deployment and serving.

## Completed work

| Homework | Topic | Status |
|----------|-------|--------|
| [Homework 1](homework-01) | Introduction to Machine Learning | Completed |
| [Homework 2](homework-02) | Machine Learning for Regression | Completed |
| [Homework 3](homework-03) | Machine Learning for Classification | Completed |
| [Homework 4](homework-04) | Evaluation Metrics for Classification | Completed |
| [Homework 5](homework-05) | Deploying Machine Learning Models | Completed |
| [Homework 6](homework-06) | Decision Trees and Ensemble Learning | Completed |
| [Homework 8](homework-08) | Neural Networks and Deep Learning | Completed |

The repository stays focused on the work that is actually present in the project.

## Repository layout

Each homework folder contains its own writeup and the files needed to reproduce that assignment. The exact contents vary by homework:

```text
homework-0X/
├── README.md
├── notebook or training script
└── supporting artifacts when needed
```

Shared datasets used across the course live in the top-level [`datasets/`](datasets) folder and are referenced by notebooks using relative paths.

## Setup

See [`SETUP.md`](SETUP.md) for the local environment setup and the pinned dependency workflow used across the course.

## Notes

- I keep notebooks when they are useful records of my work, but the course does not require every homework to use an `.ipynb` file. Scripts and checked results are appropriate for assignments such as deployment and deep learning.
- I keep generated artifacts in the repository only when they are part of the assignment. Local environments, downloaded datasets, caches, and temporary run directories stay out of Git.
- Each homework folder includes a short explanation of the objective, the approach, and the result.
- I keep the project structure consistent so the work is easy to follow and review.

## Course links

- Course repo: [DataTalksClub/machine-learning-zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- Course platform: [courses.datatalks.club/ml-zoomcamp-2026](https://courses.datatalks.club/ml-zoomcamp-2026/)
- Cohort: 2026