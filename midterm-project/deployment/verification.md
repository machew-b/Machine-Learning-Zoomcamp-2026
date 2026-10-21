# Deployment verification

The service was built and tested locally on **2026-10-21**: first as a plain Docker container, then on a local Kubernetes cluster created with [kind](https://kind.sigs.k8s.io/) (node image `kindest/node:v1.33.1`). The commands are the ones documented in sections 8 and 9 of the main README. The output below is copied from the terminal.

## 1. Docker

```powershell
docker build -t shopper-intent:v1 .
docker run --rm -p 9696:9696 shopper-intent:v1
```

The build trained the model inside the image (`RUN python download_data.py && python train.py`) and the container started `gunicorn` with two workers. In a second terminal:

```
> python predict_test.py
GET /health -> {"status":"ok"}

POST /predict -> {
  "model": "xgboost",
  "purchase_probability": 0.5626,
  "threshold": 0.2977,
  "will_purchase": true
}
```

The threshold differs slightly from the one in `reports/metrics.json` because the image retrains the model on Linux; tiny numerical differences move the F1-optimal threshold a little.

## 2. Kubernetes (kind)

```powershell
kind create cluster --name shopper
kind load docker-image shopper-intent:v1 --name shopper
kubectl apply -f deployment/k8s/
kubectl rollout status deployment/shopper-intent
kubectl get pods,svc
kubectl port-forward service/shopper-intent 8080:80
```

```
deployment.apps/shopper-intent created
service/shopper-intent created
Waiting for deployment "shopper-intent" rollout to finish: 0 of 2 updated replicas are available...
Waiting for deployment "shopper-intent" rollout to finish: 1 of 2 updated replicas are available...
deployment "shopper-intent" successfully rolled out

NAME                                  READY   STATUS    RESTARTS   AGE
pod/shopper-intent-5558bbd68b-hxkfg   1/1     Running   0          13s
pod/shopper-intent-5558bbd68b-sh9dc   1/1     Running   0          13s

NAME                     TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
service/kubernetes       ClusterIP   10.96.0.1       <none>        443/TCP   76s
service/shopper-intent   ClusterIP   10.96.192.233   <none>        80/TCP    13s

Forwarding from 127.0.0.1:8080 -> 9696
Forwarding from [::1]:8080 -> 9696
```

Testing through the port-forward:

```
> python predict_test.py http://localhost:8080
GET /health -> {"status":"ok"}

POST /predict -> {
  "model": "xgboost",
  "purchase_probability": 0.5626,
  "threshold": 0.2977,
  "will_purchase": true
}
```

Both replicas were `1/1 Running` with zero restarts, and the prediction returned through the Kubernetes service is identical to the one from the standalone container.
