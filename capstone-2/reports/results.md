### Cross-validation on the training patients (stratified, patient-grouped 5-fold)

| Model | CV log-loss (mean +/- std) | CV ROC-AUC macro | CV ROC-AUC for `<30` | CV PR-AUC for `<30` | Out-of-fold macro-F1 |
|---|---|---|---|---|---|
| logistic_regression | 0.894 +/- 0.001 | 0.661 | 0.665 | 0.208 | 0.452 |
| decision_tree | 0.906 +/- 0.002 | 0.638 | 0.631 | 0.187 | 0.432 |
| random_forest | 0.890 +/- 0.001 | 0.665 | 0.663 | 0.215 | 0.454 |
| hist_gradient_boosting (selected) | 0.884 +/- 0.001 | 0.672 | 0.672 | 0.224 | 0.463 |

### Hold-out test patients (evaluated once)

| Predictor | Log-loss | ROC-AUC macro | ROC-AUC for `<30` | PR-AUC for `<30` | Macro-F1 | Balanced accuracy | Accuracy |
|---|---|---|---|---|---|---|---|
| hist_gradient_boosting (selected model) | 0.879 | 0.678 | 0.673 | 0.229 | 0.470 | 0.468 | 0.557 |
| training class frequencies (no information) | 0.952 | 0.500 | 0.500 | 0.114 | 0.231 | 0.333 | 0.529 |

Decision weights (NO, >30, <30) = 1, 1.4, 3.25, tuned on the out-of-fold predictions of the training patients.

### Per-class results on the test patients

| Class | Precision | Recall | F1 | Test encounters | Predicted |
|---|---|---|---|---|---|
| NO | 0.661 | 0.659 | 0.660 | 10506 | 10473 |
| >30 | 0.487 | 0.508 | 0.498 | 7101 | 7406 |
| <30 | 0.268 | 0.236 | 0.251 | 2263 | 1991 |

Confusion matrix (rows = true class, columns = predicted class):

| true \ predicted | NO | >30 | <30 |
|---|---|---|---|
| NO | 6925 | 2875 | 706 |
| >30 | 2740 | 3610 | 751 |
| <30 | 808 | 921 | 534 |
