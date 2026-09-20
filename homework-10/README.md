# Homework 10: Kubernetes and Model Serving

- Status: Finished
- Instructions: [`cohorts/2026/homework/10-kubernetes`](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/main/cohorts/2026/homework/10-kubernetes) in the course repo

## Objective

Deploy the Homework 5 lead-scoring API to a local `kind` Kubernetes cluster: Deployment, ClusterIP Service, and a HorizontalPodAutoscaler with a fixed replica range.

## Files needed in VS Code

| File | Source | Purpose |
|------|--------|---------|
| `deployment.yaml` | This folder | Pod spec: image, port, resource requests, readiness probe |
| `service.yaml` | This folder | ClusterIP service exposing the deployment |
| `hpa.yaml` | This folder | Autoscaling range (1-3 replicas, CPU-based) |
| Everything from [`homework-05`](../homework-05) | Already in this repo | `pipeline.bin`, `model.py`, `predict.py`, `Dockerfile`, `q6_test.py`, etc., this homework reuses that exact artifact, just with a new image tag (`zoomcamp-model:2026-hw10`) |

Locally, this homework expects `cohorts/2026/homework/05-deployment` and `../10-kubernetes` as sibling folders. In this repo those are `homework-05/` and `homework-10/`, adjust the `cd` commands in the homework accordingly (e.g. `docker build -t zoomcamp-model:2026-hw10 .` from inside `homework-05/`, then `cd ../homework-10` for the manifests).

## Approach

Q1 needed the actual model, so rather than assume Homework 5's number still applied, I re-verified `pipeline.bin` is byte-identical to Homework 5's copy and reran the API and `q6_test.py` fresh. Q3-Q5 are Kubernetes concepts or given directly in the homework text, no cluster required to answer correctly. Q6-Q8 came from reading the actual checked-in manifests rather than guessing typical values, since this cohort's port/selector/replica numbers are pinned and version-controlled, not something to assume from memory of a typical Kubernetes tutorial.

## Outcome

| Q | Question | Answer |
|---|----------|--------|
| 1 | Local container `conversion_probability` | 0.770 (identical to Homework 5, same artifact) |
| 2 | `kind`/`kubectl` versions | Not graded, run locally and note the output |
| 3 | Smallest deployable unit | Pod |
| 4 | Default type of the `kubernetes` service | ClusterIP |
| 5 | Command to load a local image into kind | `kind load docker-image` |
| 6 | Container port in `deployment.yaml` | 9696 |
| 7 | Service selector in `service.yaml` | `app: subscription` |
| 8 | `maxReplicas` in `hpa.yaml` | 3 |
