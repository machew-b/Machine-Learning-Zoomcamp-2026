# Verification record

Real runs of the project on 1 and 5 November 2026 (Windows, PowerShell, Python 3.11.15 in a project virtual environment with the pinned requirements: numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2). Only file names are shown, never local paths.

| Step | Result |
|---|---|
| Unit tests | 85 tests, all passing (11 s). A first run of an earlier version (82 tests) had one failure: scikit-learn 1.7.2's `StratifiedGroupKFold` left the hold-out fold of a small synthetic data set with a visibly different class mix. The split was replaced by `common.GroupStratifiedKFold`, which does not depend on the library version, and its balance is now tested for several sizes and seeds. |
| `download_data.py` | `diabetic_data.csv: 101766 rows, 50 columns, 71518 patients`; `readmitted: NO 54864, >30 35545, <30 11357` (the published figures). |
| `notebook.ipynb` | Executed with `python -m nbconvert --execute --inplace` (56 cells, no cell raised an error); the executed notebook with its outputs is committed. |
| `train.py` | Completed; output below. |
| `predict.py` + `predict_test.py` | All checks passed; output below. |
| Docker build and run | Built in 147 s (about 100 s for the download and training inside the image); the container started gunicorn with two workers and answered `/health`; `predict_test.py` against it: all 23 checks passed; `GET /` reported `hist_gradient_boosting`, weights 1 / 1.4 / 3.25, 99,343 rows (see below). |
| Kubernetes (kind) | Two replicas `1/1 Running`, 0 restarts, deployment `2/2`, ClusterIP service on port 80; `predict_test.py` through `kubectl port-forward`: all checks passed (see below). `kind load` needed the workaround described below. |

## `python train.py`

```
Loaded 101766 encounters; dropped 0 duplicates and 2423 expired / hospice discharges -> 99343 encounters of 69990 patients.
Patient-aware split: 79473 training encounters (55991 patients), 19870 test encounters (13999 patients).
  logistic_regression      CV log-loss 0.8937 +/- 0.0010   AUC macro 0.661   AUC <30 0.665   OOF macro-F1 0.452   (13s)
  decision_tree            CV log-loss 0.9057 +/- 0.0017   AUC macro 0.638   AUC <30 0.631   OOF macro-F1 0.432   (7s)
  random_forest            CV log-loss 0.8902 +/- 0.0013   AUC macro 0.665   AUC <30 0.663   OOF macro-F1 0.454   (14s)
  hist_gradient_boosting   CV log-loss 0.8838 +/- 0.0013   AUC macro 0.672   AUC <30 0.672   OOF macro-F1 0.463   (21s)
Selected model (lowest CV log-loss): hist_gradient_boosting; decision weights [1.0, 1.4, 3.25]
Test patients: log-loss 0.8790  AUC macro 0.678  AUC <30 0.673  macro-F1 0.470  recall <30 0.236
  reference (class frequencies): log-loss 0.9521  macro-F1 0.231
Saved model.bin (refit on all 99343 encounters), reports/metrics.json and reports/results.md in 68s.
```

The full tables are in `reports/results.md` and `reports/metrics.json`.

## `python predict_test.py` against `python predict.py`

```
OK   GET /health  -> {'status': 'ok', 'model': 'hist_gradient_boosting'}
OK   GET /  -> model hist_gradient_boosting
OK   POST /predict (sample_request.json)  -> >30 | P(none)=0.324 P(>30)=0.501 P(<30)=0.175
OK   POST /predict low-utilisation patient  -> NO | P(none)=0.774 P(>30)=0.181 P(<30)=0.045
OK   POST /predict high-utilisation patient  -> <30 | P(none)=0.181 P(>30)=0.448 P(<30)=0.370
NOTE sanity: early-readmission risk of the high-utilisation patient > the low-utilisation one (0.370 vs 0.045); more prior inpatient visits should raise the risk
OK   POST /predict with optional fields omitted  -> >30 | P(none)=0.283 P(>30)=0.536 P(<30)=0.181
OK   POST /predict unusual values give warnings, not errors  -> 3 warning(s)
OK   POST /predict_batch (5 records)  -> early risk by prior inpatient visits: 0.103, 0.136, 0.175, 0.202, 0.238
OK   invalid input rejected: ... (12 cases: age label, expired and hospice discharge, missing field, unknown field,
     invalid ICD-9 code, unknown drug, invalid drug status, negative count, number sent as text, boolean sent as number,
     lab result not allowed)  -> 400 with the reason
OK   invalid input rejected: not JSON  -> 400
OK   empty batch rejected  -> 400
OK   unknown route returns 404  -> 404

All checks passed.
```

(The twelve rejection lines are shortened here; each printed `OK` with the HTTP status 400 and the reason.)

## Docker

```
docker build -t readmission-risk:1.0 .      -> [+] Building 147.1s (12/12) FINISHED   (step 6/7: python download_data.py && python train.py, 100.5 s)
docker run --rm -p 9696:9696 readmission-risk:1.0
[INFO] Starting gunicorn 23.0.0
[INFO] Listening at: http://0.0.0.0:9696 (1)
[INFO] Using worker: sync
[INFO] Booting worker with pid: 37
[INFO] Booting worker with pid: 38
"GET /health HTTP/1.1" 200 49     (repeated every 30 s by the image's health check)
```

`python predict_test.py http://localhost:9696` printed the same 23 `OK` lines as the local run and `All checks passed.` `GET /` returned, among other fields: `model_name` `hist_gradient_boosting`, `decision_weights` [1.0, 1.4, 3.25], `fit_rows` 99343, `fit_patients` 69990, `params_status` `tuned`, `holdout_test` log-loss 0.879 and ROC-AUC for `<30` 0.673, `training_ranges` for the eight counts (for example `time_in_hospital` 1 to 14, `number_inpatient` 0 to 21). These are the values of the local training: training inside the image is reproducible.

## Kubernetes (kind)

`kind load docker-image readmission-risk:1.0 --name readmission` failed with `ERROR: failed to load image ... ctr: rpc error: code = Unavailable desc = error reading from server: EOF`, and the pods stayed in `ImagePullBackOff`. Rebuilding the image without provenance and SBOM data and using `kind load image-archive` failed in the same way, while the node had 927 GB of free disk space. Copying the archive into the node and importing it there worked (`docker cp` followed by `docker exec ... ctr -n k8s.io images import`; `crictl images` then listed `docker.io/library/readmission-risk 1.0`, 161 MB). After that:

```
kubectl apply -f deployment/k8s/
deployment.apps/readmission-risk created
service/readmission-risk created
Waiting for deployment "readmission-risk" rollout to finish: 0 of 2 updated replicas are available...
Waiting for deployment "readmission-risk" rollout to finish: 1 of 2 updated replicas are available...
deployment "readmission-risk" successfully rolled out

kubectl get pods
NAME                               READY   STATUS    RESTARTS   AGE
readmission-risk-56f86b7978-k4x65  1/1     Running   0          7s
readmission-risk-56f86b7978-n5x46  1/1     Running   0          7s

kubectl get deployment,service,pods
deployment.apps/readmission-risk   2/2   2   2   101s
service/kubernetes                 ClusterIP   10.96.0.1        <none>   443/TCP   3m55s
service/readmission-risk           ClusterIP   10.96.239.153    <none>   80/TCP    101s

kubectl port-forward service/readmission-risk 8080:80
Forwarding from 127.0.0.1:8080 -> 9696
python predict_test.py http://localhost:8080      -> 23 OK lines, All checks passed.
kind delete cluster --name readmission            -> Deleted nodes: ["readmission-control-plane"]
```

Screenshots: `images/k8s-pods.png`, `images/k8s-resources.png`, `images/k8s-predict-test.png` (Kubernetes) and `images/train-output.png`, `images/predict-test-local.png` (local run).

## Not run

The Google Cloud Run deployment is documented in the README but was not run (it needs a Google Cloud project with billing enabled).

## Development-sandbox checks (before the real run)

The code was first written and tested without internet access, Docker or Kubernetes, on synthetic data in the real file's format (Python 3.12, numpy 2.4, pandas 3.0, scikit-learn 1.8): the unit tests, `train.py`, the notebook executed cell by cell as a script, and the service. These runs found and fixed a bug (omitting the optional `medications` field returned HTTP 500) before the first real run; their numbers are meaningless and are not used anywhere.
