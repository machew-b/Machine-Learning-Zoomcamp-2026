"""Train the purchase-intent model and save it to ``model.bin``.

Steps (mirrors the notebook):
  1. load + clean the data, hold out a stratified 20 % test set
  2. 5-fold cross-validation of every candidate model on the training part
  3. pick the model with the best ROC-AUC
  4. choose the decision threshold that maximises F1 on out-of-fold predictions
  5. refit on the full training part, evaluate once on the test set
  6. save the pipeline + threshold (``model.bin``) and the numbers (``reports/metrics.json``)

Hyper-parameters come from ``params.json`` (written by the notebook's tuning step).

Usage:
    python train.py
    python train.py --data data/online_shoppers_intention.csv --model-out model.bin
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--data", default=str(common.DATA_PATH), help="path to the CSV file")
    parser.add_argument("--params", default=str(common.PARAMS_PATH), help="hyper-parameters (JSON)")
    parser.add_argument("--model-out", default=str(common.MODEL_PATH), help="where to save the model")
    parser.add_argument("--reports-dir", default=str(common.REPORTS_DIR), help="where to save metrics.json")
    parser.add_argument("--folds", type=int, default=5, help="number of cross-validation folds")
    parser.add_argument("--seed", type=int, default=common.SEED, help="random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. data --------------------------------------------------------------
    df = common.load_dataset(args.data)
    X, y = df[common.RAW_FEATURES], df[common.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed
    )
    print(f"{len(df):,} unique sessions | train {len(X_train):,} | test {len(X_test):,} "
          f"| purchase rate {y.mean():.1%}")

    # 2. cross-validate every candidate -------------------------------------
    params = common.load_params(args.params)
    cv_results = {}
    for name in common.MODEL_NAMES:
        start = time.time()
        pipeline = common.build_pipeline(name, params.get(name), seed=args.seed)
        cv_results[name] = common.cv_scores(pipeline, X_train, y_train, folds=args.folds, seed=args.seed)
        r = cv_results[name]
        print(f"  {name:<20} ROC-AUC {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} "
              f"| PR-AUC {r['pr_auc_mean']:.4f}  ({time.time() - start:.0f}s)")

    # 3. select the best model ----------------------------------------------
    best_name = max(cv_results, key=lambda n: cv_results[n]["roc_auc_mean"])
    print(f"Selected model: {best_name}")
    pipeline = common.build_pipeline(best_name, params.get(best_name), seed=args.seed)

    # 4. decision threshold from out-of-fold predictions ---------------------
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    threshold = common.best_f1_threshold(y_train, oof)

    # 5. final fit + one look at the test set -------------------------------
    pipeline.fit(X_train, y_train)
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    test_tuned = common.evaluate(y_test, proba_test, threshold)
    test_default = common.evaluate(y_test, proba_test, 0.5)
    print(f"Test ROC-AUC {test_tuned['roc_auc']:.4f} | PR-AUC {test_tuned['pr_auc']:.4f}")
    print(f"At threshold {threshold:.3f}: precision {test_tuned['precision']:.3f} "
          f"recall {test_tuned['recall']:.3f} F1 {test_tuned['f1']:.3f}")

    # 6. persist ------------------------------------------------------------
    bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "model_name": best_name,
        "features": common.RAW_FEATURES,
        "test_metrics": test_tuned,
        "versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "selected_model": best_name,
        "hyperparameters": params.get(best_name, {}),
        "cross_validation_on_train": cv_results,
        "test_at_tuned_threshold": test_tuned,
        "test_at_threshold_0.5": test_default,
        "n_sessions": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "purchase_rate": float(y.mean()),
        "seed": args.seed,
        "versions": bundle["versions"],
    }
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(common.to_builtin(metrics), f, indent=2)
        f.write("\n")
    print(f"Saved model to {model_path} and metrics to {reports_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
