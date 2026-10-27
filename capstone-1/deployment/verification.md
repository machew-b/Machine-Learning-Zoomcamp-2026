# Capstone 1: verification

This file records the runs that were used to check the project end to end. Everything was run on one Windows machine (PowerShell), with Python 3.11 and the pinned package versions (numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2, matplotlib 3.10.0, seaborn 0.13.1, Flask 3.1.3), Docker Desktop 4.91.0 (Docker Engine 29.8.0) and kind with the node image `kindest/node:v1.33.1`. The output below is copied from the terminal; empty or unrelated lines are left out.

The same smoke test, [`predict_test.py`](../predict_test.py), was run against the service in three places: locally, in the Docker container and in the Kubernetes cluster. All three return identical forecasts.

## 1. Environment and unit tests

```text
> python -c "import sys, numpy, pandas, sklearn, flask; print(sys.version.split()[0], numpy.__version__, pandas.__version__, sklearn.__version__, flask.__version__)"
3.11.15 2.3.3 2.3.2 1.7.2 3.1.3

> python -m unittest discover -s tests
.....................................................................................
Ran 85 tests in 3.318s

OK
```

## 2. Data, notebook and training

```text
> python download_data.py
Downloading the dataset from the UCI Machine Learning Repository ...
Saved SeoulBikeData.csv.
SeoulBikeData.csv: 8760 rows, 14 columns, 2017-12-01 to 2018-11-30
Data licence: CC BY 4.0 (UCI Machine Learning Repository, doi:10.24432/C5F62R).

> python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook.ipynb
[NbConvertApp] Writing 2056513 bytes to notebook.ipynb
```

At the end of the notebook run, joblib printed `KeyError ... joblib_memmapping_folder` messages. They come from its temporary-folder cleanup on Windows and do not affect the result.

`params.json` was rewritten by the notebook with `"status": "tuned"`. Then:

```text
> python train.py
Loaded 8760 rows; dropped 0 duplicates and 295 hours with the system switched off -> 8465 rows.
Chronological split: 6768 training rows (until 2018-09-10), 1697 test rows (from 2018-09-11).
  ridge                    CV RMSE   354.1 +/-  17.4   MAE  255.1   R2 0.694   (0s)
  decision_tree            CV RMSE   241.9 +/-  21.4   MAE  149.0   R2 0.858   (0s)
  random_forest            CV RMSE   197.4 +/-  21.0   MAE  122.1   R2 0.905   (14s)
  hist_gradient_boosting   CV RMSE   187.7 +/-  25.3   MAE  114.5   R2 0.914   (4s)
Selected model (lowest CV RMSE): hist_gradient_boosting
Test period: RMSE 320.8  MAE 242.9  R2 0.713  WAPE 27.1 %
  reference 'same hour and day type' average: RMSE 434.7  MAE 317.7  R2 0.473
Saved model.bin (refit on all 8465 rows), reports/metrics.json and reports/results.md in 19s.
```

## 3. Local service

```text
> python predict.py          (terminal 1)
> python predict_test.py     (terminal 2)
OK   GET /health  -> {'status': 'ok', 'model': 'hist_gradient_boosting'}
OK   GET /  -> model hist_gradient_boosting
OK   POST /predict (sample_request.json)  -> 2902.7 bikes per hour
OK   POST /predict hour=03 (weekday)  -> 332.6
OK   POST /predict hour=08 (weekday)  -> 1875.1
OK   POST /predict hour=13 (weekday)  -> 985.6
OK   POST /predict hour=18 (weekday)  -> 2399.2
NOTE sanity: 18:00 forecast > 03:00 forecast (a commuting evening should be busier than the middle of the night)
OK   POST /predict rainy evening  -> 545.7
OK   POST /predict Sunday afternoon  -> 1329.9
OK   POST /predict_batch (24 hours)  -> peak hour 18
OK   invalid input rejected: hour out of range (24)  -> 400 ['hour must be between 0 and 23, got 24']
OK   invalid input rejected: missing field  -> 400 ['missing field(s): temperature_c']
OK   invalid input rejected: unknown field  -> 400 ['unknown field(s): season']
OK   invalid input rejected: invalid date  -> 400 ["date must be a valid calendar date in the format YYYY-MM-DD, got '2018-02-30'"]
OK   invalid input rejected: humidity above 100  -> 400 ['humidity_pct must be between 0.0 and 100.0 %, got 120.0']
OK   invalid input rejected: number sent as text  -> 400 ['temperature_c must be a number']
OK   invalid input rejected: boolean expected  -> 400 ['holiday must be true or false']
OK   invalid input rejected: negative rainfall  -> 400 ['rainfall_mm must be between 0.0 and 300.0 mm, got -1.0']
OK   invalid input rejected: not JSON  -> 400
OK   unknown route returns 404  -> 404

All checks passed.
```

## 4. Docker

The image downloads the data and trains the model during the build.

```text
> docker build -t seoul-bike-demand:1.0 .
[+] Building 67.9s (13/13) FINISHED
 => [1/7] FROM docker.io/library/python:3.11-slim-bookworm
 => [4/7] RUN pip install -r requirements.txt                                   19.3s
 => [5/7] COPY common.py download_data.py train.py predict.py params.json ./     0.2s
 => [6/7] RUN python download_data.py && python train.py                        27.8s
 => [7/7] RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app   0.4s
 => naming to docker.io/library/seoul-bike-demand:1.0

> docker run -d --rm --name seoul-bike-test -p 9696:9696 seoul-bike-demand:1.0
> docker ps
CONTAINER ID   IMAGE                   COMMAND                  CREATED          STATUS                    PORTS                                         NAMES
b96c6e911145   seoul-bike-demand:1.0   "sh -c 'exec gunicor…"   46 seconds ago   Up 45 seconds (healthy)   0.0.0.0:9696->9696/tcp, [::]:9696->9696/tcp   seoul-bike-test

> python predict_test.py http://localhost:9696
OK   GET /health  -> {'status': 'ok', 'model': 'hist_gradient_boosting'}
OK   POST /predict (sample_request.json)  -> 2902.7 bikes per hour
OK   POST /predict hour=03 (weekday)  -> 332.6
OK   POST /predict hour=08 (weekday)  -> 1875.1
OK   POST /predict hour=13 (weekday)  -> 985.6
OK   POST /predict hour=18 (weekday)  -> 2399.2
...
All checks passed.

> docker logs seoul-bike-test --tail 15
172.17.0.1 - - [27/Oct/2026:03:40:19 +0000] "POST /predict HTTP/1.1" 200 99 "-" "Python-urllib/3.11"
172.17.0.1 - - [27/Oct/2026:03:40:19 +0000] "POST /predict_batch HTTP/1.1" 200 1359 "-" "Python-urllib/3.11"
172.17.0.1 - - [27/Oct/2026:03:40:19 +0000] "POST /predict HTTP/1.1" 400 78 "-" "Python-urllib/3.11"
...
172.17.0.1 - - [27/Oct/2026:03:40:19 +0000] "GET /nothing-here HTTP/1.1" 404 22 "-" "Python-urllib/3.11"

> docker stop seoul-bike-test
```

(The other containers in the `docker ps` listing, which belong to other projects, are left out.)

## 5. Kubernetes (local kind cluster)

```text
> kind create cluster --name seoul-bike
Creating cluster "seoul-bike" ...
 ✓ Ensuring node image (kindest/node:v1.33.1)
 ✓ Preparing nodes
 ✓ Writing configuration
 ✓ Starting control-plane
 ✓ Installing CNI
 ✓ Installing StorageClass
Set kubectl context to "kind-seoul-bike"

> kubectl config current-context
kind-seoul-bike

> kind load docker-image seoul-bike-demand:1.0 --name seoul-bike
Image: "seoul-bike-demand:1.0" with ID "sha256:a81117df79101a407248b2780987699c3500671c20982ec94fff82cc5259afde" not yet present on node "seoul-bike-control-plane", loading...

> kubectl apply -f deployment/k8s/
deployment.apps/seoul-bike-demand created
service/seoul-bike-demand created

> kubectl rollout status deployment/seoul-bike-demand --timeout=180s
Waiting for deployment "seoul-bike-demand" rollout to finish: 0 of 2 updated replicas are available...
Waiting for deployment "seoul-bike-demand" rollout to finish: 1 of 2 updated replicas are available...
deployment "seoul-bike-demand" successfully rolled out

> kubectl get pods
NAME                                 READY   STATUS    RESTARTS   AGE
seoul-bike-demand-675784bdcb-bzs9n   1/1     Running   0          6s
seoul-bike-demand-675784bdcb-f2wph   1/1     Running   0          6s

> kubectl get svc seoul-bike-demand
NAME                TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
seoul-bike-demand   ClusterIP   10.96.236.141   <none>        80/TCP    7s

> kubectl port-forward service/seoul-bike-demand 8080:80
Forwarding from 127.0.0.1:8080 -> 9696
Forwarding from [::1]:8080 -> 9696
```

In a second terminal:

```text
> python predict_test.py http://localhost:8080
OK   GET /health  -> {'status': 'ok', 'model': 'hist_gradient_boosting'}
OK   POST /predict (sample_request.json)  -> 2902.7 bikes per hour
OK   POST /predict hour=03 (weekday)  -> 332.6
OK   POST /predict hour=08 (weekday)  -> 1875.1
OK   POST /predict hour=13 (weekday)  -> 985.6
OK   POST /predict hour=18 (weekday)  -> 2399.2
...
All checks passed.
```

About five minutes later the pods were still ready, without restarts:

```text
> kubectl get pods
NAME                                 READY   STATUS    RESTARTS   AGE
seoul-bike-demand-675784bdcb-bzs9n   1/1     Running   0          5m18s
seoul-bike-demand-675784bdcb-f2wph   1/1     Running   0          5m18s
```

Screenshots: [`images/k8s-pods.png`](../images/k8s-pods.png) and [`images/k8s-predict.png`](../images/k8s-predict.png).

## What this shows

- The unit tests, the notebook and `train.py` work with the pinned versions on the real data.
- The model returns the same forecasts when it is served locally, in the Docker container and through Kubernetes (two replicas, readiness and liveness probes on `/health`).
- Invalid requests are rejected with HTTP 400 and a reason, and unknown routes return 404, in all three places.

To repeat any step, follow [Run it yourself](../README.md#run-it-yourself), [Docker](../README.md#docker) and [Kubernetes](../README.md#kubernetes-local-kind-cluster) in the project README.
