# Verification record

**Status: the real run has not been done yet.** This file is completed from real outputs after the project has been run on the real data (unit tests, data download, notebook, `train.py`, local service, Docker, Kubernetes). Nothing below is claimed before it has been observed.

## What was checked while the project was written (development sandbox, synthetic data in the real file format)

The sandbox had no internet access, no Docker and no Kubernetes, and had newer libraries than the pinned ones (Python 3.12, numpy 2.4, pandas 3.0, scikit-learn 1.8, Flask 3.1), so the following was run on **synthetic data only**; its numbers are meaningless and are never used as results.

- `python -m unittest discover -s tests`: 85 tests, all passing in the sandbox. A first run on a different machine (Python 3.11 with the pinned scikit-learn 1.7.2) showed that scikit-learn's `StratifiedGroupKFold` left the hold-out fold with a visibly different class mix on a small synthetic data set; the split was therefore replaced by `common.GroupStratifiedKFold`, which does not depend on the library version, and its balance is now tested for several sizes and seeds.
- `train.py` end to end on 8,000 synthetic encounters (cross-validation of four models, selection, test evaluation, refit, `model.bin`, reports).
- The whole notebook, executed cell by cell as a script (with IPython stubbed): once with a reduced search budget and once with the full budget, on 8,000 synthetic encounters.
- The Flask service (development server) with `predict_test.py`: all checks passed.

## What could not be run there

- The real dataset (download, schema, results), with the pinned versions (numpy 2.3.3, pandas 2.3.2, scikit-learn 1.7.2).
- gunicorn, the Docker build and run, the Kubernetes deployment, the Cloud Run deployment.

## Real run

To be filled in: environment versions, test output, download output, notebook execution, `train.py` output and `reports/results.md`, `predict_test.py` against the local service, the Docker container and the kind cluster.
