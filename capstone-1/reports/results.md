### Cross-validation on the training period (blocked 5-fold, bikes per hour)

| Model | CV RMSE (mean +/- std) | CV MAE | CV R2 |
|---|---|---|---|
| ridge | 354.1 +/- 17.4 | 255.1 | 0.694 |
| decision_tree | 241.9 +/- 21.4 | 149.0 | 0.858 |
| random_forest | 197.4 +/- 21.0 | 122.1 | 0.905 |
| hist_gradient_boosting (selected) | 187.7 +/- 25.3 | 114.5 | 0.914 |

### Hold-out test period (evaluated once)

| Predictor | RMSE | MAE | R2 | WAPE |
|---|---|---|---|---|
| hist_gradient_boosting (selected model) | 320.8 | 242.9 | 0.713 | 27.1 % |
| average of the same hour and day type | 434.7 | 317.7 | 0.473 | 35.5 % |
| overall training average | 633.8 | 490.9 | -0.120 | 54.9 % |
