#!/usr/bin/env python
"""Train the final model and write model.bin plus reports/metrics.json.

    python train.py

Steps
1. load and clean the data (common.py);
2. patient-aware split: about 20 % of the patients (all of their encounters) form the hold-out test set;
3. stratified, patient-grouped 5-fold cross-validation on the training patients for all four candidate
   models, with the hyper-parameters in params.json (written by the notebook);
4. select the model with the lowest cross-validated log-loss (the test set plays no role here);
5. tune the decision weights of the selected model on its out-of-fold predictions (maximum macro-F1);
6. evaluate the selected model once on the test set, next to a naive reference predictor;
7. refit the selected pipeline on ALL cleaned rows and save it as model.bin.
   The reported test metrics belong to the model trained on the training patients only; the saved
   model has seen every patient, so it cannot be scored on the test patients any more.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import common

HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the hospital-readmission model.")
    parser.add_argument("--data", default=str(HERE / "data" / common.CSV_NAME), help="path of diabetic_data.csv")
    parser.add_argument("--params", default=str(HERE / "params.json"), help="hyper-parameters written by the notebook")
    parser.add_argument("--model-out", default=str(HERE / "model.bin"), help="where to save the model bundle")
    parser.add_argument("--reports-dir", default=str(HERE / "reports"), help="where to write metrics.json / results.md")
    parser.add_argument("--n-jobs", type=int, default=-1, help="CPU cores for the random forest (-1 = all)")
    return parser.parse_args(argv)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def results_markdown(metrics: dict) -> str:
    """Markdown tables that can be pasted into the README."""
    cv = metrics["cv"]
    selected = metrics["selected_model"]
    test = metrics["test"]["selected_model"]
    reference = metrics["test"]["baseline_class_frequencies"]
    labels = common.CLASS_LABELS
    lines = [
        f"### Cross-validation on the training patients (stratified, patient-grouped {common.N_SPLITS}-fold)",
        "",
        "| Model | CV log-loss (mean +/- std) | CV ROC-AUC macro | CV ROC-AUC for `<30` | CV PR-AUC for `<30` | Out-of-fold macro-F1 |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in cv.items():
        label = f"{name} (selected)" if name == selected else name
        lines.append(
            f"| {label} | {_fmt(s['log_loss_mean'])} +/- {_fmt(s['log_loss_std'])} | {_fmt(s['roc_auc_macro_ovr_mean'])} "
            f"| {_fmt(s['roc_auc_early_mean'])} | {_fmt(s['pr_auc_early_mean'])} | {_fmt(s['oof_macro_f1'])} |"
        )
    lines += [
        "",
        "### Hold-out test patients (evaluated once)",
        "",
        "| Predictor | Log-loss | ROC-AUC macro | ROC-AUC for `<30` | PR-AUC for `<30` | Macro-F1 | Balanced accuracy | Accuracy |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, m in ((f"{selected} (selected model)", test), ("training class frequencies (no information)", reference)):
        lines.append(
            f"| {label} | {_fmt(m['log_loss'])} | {_fmt(m['roc_auc_macro_ovr'])} | {_fmt(m['roc_auc_early'])} "
            f"| {_fmt(m['pr_auc_early'])} | {_fmt(m['macro_f1'])} | {_fmt(m['balanced_accuracy'])} | {_fmt(m['accuracy'])} |"
        )
    weights = test["weights"]
    lines += [
        "",
        f"Decision weights (NO, >30, <30) = {weights[0]:g}, {weights[1]:g}, {weights[2]:g}, tuned on the out-of-fold predictions of the training patients.",
        "",
        "### Per-class results on the test patients",
        "",
        "| Class | Precision | Recall | F1 | Test encounters | Predicted |",
        "|---|---|---|---|---|---|",
    ]
    for label in labels:
        c = test["per_class"][label]
        lines.append(
            f"| {label} | {_fmt(c['precision'])} | {_fmt(c['recall'])} | {_fmt(c['f1'])} | {c['support']} | {c['predicted']} |"
        )
    lines += ["", "Confusion matrix (rows = true class, columns = predicted class):", "", "| true \\ predicted | " + " | ".join(labels) + " |",
              "|---|" + "---|" * len(labels)]
    for label, row in zip(labels, test["confusion_matrix"]):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: {data_path.name} not found. Run `python download_data.py` first.")
        return 1

    raw = common.load_raw(data_path)
    data, report = common.clean_with_report(raw)
    print(
        f"Loaded {report['rows_raw']} encounters; dropped {report['duplicate_rows_dropped']} duplicates and "
        f"{report['expired_or_hospice_dropped']} expired / hospice discharges -> {report['rows_clean']} encounters "
        f"of {report['patients_clean']} patients."
    )

    X = data[common.INPUT_COLUMNS]
    y = data[common.TARGET]
    groups = data[common.GROUP_COLUMN].to_numpy()
    train_mask, test_mask = common.grouped_stratified_split(y, groups)
    X_train, y_train, g_train = X.loc[train_mask], y.loc[train_mask], groups[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    print(
        f"Patient-aware split: {len(X_train)} training encounters ({len(np.unique(g_train))} patients), "
        f"{len(X_test)} test encounters ({len(np.unique(groups[test_mask]))} patients)."
    )

    params, params_meta = common.load_params(args.params)
    if params_meta.get("status") != "tuned":
        print("WARNING: params.json holds default (untuned) hyper-parameters. Run the notebook to tune them.")

    cv_results = {}
    for name in common.MODEL_NAMES:
        t0 = time.time()
        pipeline = common.build_pipeline(name, params[name], n_jobs=args.n_jobs)
        cv = common.cross_validate_model(pipeline, X_train, y_train, g_train)
        weights, oof_f1 = common.tune_decision_weights(y_train, cv["oof_proba"])
        cv_results[name] = {**cv["summary"], "oof_macro_f1": oof_f1, "decision_weights": weights}
        s = cv["summary"]
        print(
            f"  {name:<24s} CV log-loss {s['log_loss_mean']:.4f} +/- {s['log_loss_std']:.4f}   "
            f"AUC macro {s['roc_auc_macro_ovr_mean']:.3f}   AUC <30 {s['roc_auc_early_mean']:.3f}   "
            f"OOF macro-F1 {oof_f1:.3f}   ({time.time() - t0:.0f}s)"
        )

    best = min(cv_results, key=lambda n: cv_results[n]["log_loss_mean"])
    weights = cv_results[best]["decision_weights"]
    print(f"Selected model (lowest CV log-loss): {best}; decision weights {weights}")

    candidate = common.build_pipeline(best, params[best], n_jobs=args.n_jobs).fit(X_train, y_train)
    proba_test = common.predict_proba(candidate, X_test)
    test_metrics = common.classification_metrics(y_test, proba_test, weights)
    reference = common.classification_metrics(
        y_test, common.prior_probabilities(y_train, len(y_test)), [1.0, 1.0, 1.0]
    )
    print(
        f"Test patients: log-loss {test_metrics['log_loss']:.4f}  AUC macro {test_metrics['roc_auc_macro_ovr']:.3f}  "
        f"AUC <30 {test_metrics['roc_auc_early']:.3f}  macro-F1 {test_metrics['macro_f1']:.3f}  "
        f"recall <30 {test_metrics['per_class']['<30']['recall']:.3f}"
    )
    print(f"  reference (class frequencies): log-loss {reference['log_loss']:.4f}  macro-F1 {reference['macro_f1']:.3f}")

    final = common.build_pipeline(best, params[best], n_jobs=args.n_jobs).fit(X, y)
    common.make_serving_ready(final)
    base_rate = np.bincount(y, minlength=common.N_CLASSES) / len(y)
    metadata = {
        "model_name": best,
        "params": params[best],
        "classes": common.CLASS_LABELS,
        "class_keys": common.CLASS_KEYS,
        "decision_weights": weights,
        "base_rates": [float(v) for v in base_rate],
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fit_rows": int(len(data)),
        "fit_patients": int(len(np.unique(groups))),
        "training_ranges": common.training_ranges(X),
        "training_categories": common.training_categories(X),
        "cv": cv_results[best],
        "holdout_test": {
            "rows": int(len(X_test)),
            **{k: v for k, v in test_metrics.items() if k not in ("per_class", "confusion_matrix")},
            "recall_early": test_metrics["per_class"]["<30"]["recall"],
            "precision_early": test_metrics["per_class"]["<30"]["precision"],
        },
        "params_status": params_meta.get("status", "unknown"),
        "versions": common.library_versions(),
    }
    model_out = Path(args.model_out)
    common.save_bundle(model_out, final, common.to_builtin(metadata))

    metrics = common.to_builtin(
        {
            "seed": common.RANDOM_STATE,
            "data": {
                **report,
                "train_rows": int(len(X_train)),
                "train_patients": int(len(np.unique(g_train))),
                "test_rows": int(len(X_test)),
                "test_patients": int(len(np.unique(groups[test_mask]))),
                "class_share_train": [float(v) for v in np.bincount(y_train, minlength=3) / len(y_train)],
                "class_share_test": [float(v) for v in np.bincount(y_test, minlength=3) / len(y_test)],
            },
            "validation": {
                "test_split": "StratifiedGroupKFold(5), one fold = about 20 % of the patients, grouped by patient_nbr",
                "cv": f"StratifiedGroupKFold({common.N_SPLITS}) grouped by patient_nbr, training patients only",
                "selection": "lowest mean cross-validated log-loss",
                "decision_weights": "grid search of the class weights maximising out-of-fold macro-F1",
            },
            "params_status": params_meta.get("status", "unknown"),
            "cv": cv_results,
            "selected_model": best,
            "selected_params": params[best],
            "test": {"selected_model": test_metrics, "baseline_class_frequencies": reference},
            "final_model": {
                "refit_on": "all cleaned encounters (training + test patients)",
                "rows": int(len(data)),
                "note": "test metrics above were measured on a model trained on the training patients only",
            },
            "versions": common.library_versions(),
        }
    )
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_text(reports_dir / "metrics.json", json.dumps(metrics, indent=2) + "\n")
    write_text(reports_dir / "results.md", results_markdown(metrics))
    print(
        f"Saved {model_out.name} (refit on all {len(data)} encounters), {reports_dir.name}/metrics.json and "
        f"{reports_dir.name}/results.md in {time.time() - started:.0f}s."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
