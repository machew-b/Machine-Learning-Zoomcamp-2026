#!/usr/bin/env python
"""Train the final model and write model.bin plus reports/metrics.json.

    python train.py

Steps
1. load and clean the data (common.py);
2. chronological split: the last 20 % of the calendar days are the hold-out test set;
3. blocked 5-fold cross-validation on the training period for all four candidate models, with
   the hyper-parameters in params.json (written by the notebook);
4. select the model with the lowest cross-validated RMSE (the test set plays no role here);
5. evaluate the selected model once on the test set, next to two naive reference predictors;
6. refit the selected pipeline on ALL cleaned rows and save it as model.bin.
   The reported test metrics belong to the model trained on the training period only; the saved
   model has seen every day, so it cannot be scored on the test period any more.
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
    parser = argparse.ArgumentParser(description="Train the Seoul bike-demand model.")
    parser.add_argument("--data", default=str(HERE / "data" / common.CSV_NAME), help="path of SeoulBikeData.csv")
    parser.add_argument("--params", default=str(HERE / "params.json"), help="hyper-parameters written by the notebook")
    parser.add_argument("--model-out", default=str(HERE / "model.bin"), help="where to save the model bundle")
    parser.add_argument("--reports-dir", default=str(HERE / "reports"), help="where to write metrics.json / results.md")
    return parser.parse_args(argv)


def _fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def results_markdown(metrics: dict) -> str:
    """Markdown tables that can be pasted into the README."""
    cv = metrics["cv"]
    selected = metrics["selected_model"]
    lines = [
        "### Cross-validation on the training period (blocked 5-fold, bikes per hour)",
        "",
        "| Model | CV RMSE (mean +/- std) | CV MAE | CV R2 |",
        "|---|---|---|---|",
    ]
    for name, s in cv.items():
        label = f"{name} (selected)" if name == selected else name
        lines.append(
            f"| {label} | {_fmt(s['rmse_mean'])} +/- {_fmt(s['rmse_std'])} | {_fmt(s['mae_mean'])} | {_fmt(s['r2_mean'], 3)} |"
        )
    lines += [
        "",
        "### Hold-out test period (evaluated once)",
        "",
        "| Predictor | RMSE | MAE | R2 | WAPE |",
        "|---|---|---|---|---|",
    ]
    for label, m in (
        (f"{selected} (selected model)", metrics["test"]["selected_model"]),
        ("average of the same hour and day type", metrics["test"]["baseline_hour_daytype_mean"]),
        ("overall training average", metrics["test"]["baseline_training_mean"]),
    ):
        lines.append(
            f"| {label} | {_fmt(m['rmse'])} | {_fmt(m['mae'])} | {_fmt(m['r2'], 3)} | {_fmt(100 * m['wape'])} % |"
        )
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
        f"Loaded {report['rows_raw']} rows; dropped {report['duplicate_rows_dropped']} duplicates and "
        f"{report['non_functioning_rows_dropped']} hours with the system switched off -> {report['rows_clean']} rows."
    )

    X = data[common.INPUT_COLUMNS]
    y = data[common.TARGET]
    train_mask, test_mask, cutoff = common.time_split_masks(data["date"])
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    groups = common.block_groups(data.loc[train_mask, "date"])
    print(
        f"Chronological split: {len(X_train)} training rows (until {(cutoff - np.timedelta64(1, 'D')).date()}), "
        f"{len(X_test)} test rows (from {cutoff.date()})."
    )

    params, params_meta = common.load_params(args.params)
    if params_meta.get("status") != "tuned":
        print("WARNING: params.json holds default (untuned) hyper-parameters. Run the notebook to tune them.")

    cv_results = {}
    for name in common.MODEL_NAMES:
        t0 = time.time()
        cv = common.cross_validate_model(common.build_pipeline(name, params[name]), X_train, y_train, groups)
        cv_results[name] = cv["summary"]
        s = cv["summary"]
        print(
            f"  {name:<24s} CV RMSE {s['rmse_mean']:7.1f} +/- {s['rmse_std']:5.1f}   "
            f"MAE {s['mae_mean']:6.1f}   R2 {s['r2_mean']:.3f}   ({time.time() - t0:.0f}s)"
        )

    best = min(cv_results, key=lambda n: cv_results[n]["rmse_mean"])
    print(f"Selected model (lowest CV RMSE): {best}")

    candidate = common.build_pipeline(best, params[best]).fit(X_train, y_train)
    test_metrics = common.regression_metrics(y_test, common.predict_counts(candidate, X_test))
    profile_metrics = common.regression_metrics(y_test, common.profile_baseline(data.loc[train_mask], data.loc[test_mask]))
    mean_metrics = common.regression_metrics(y_test, np.full(len(y_test), float(y_train.mean())))
    print(
        f"Test period: RMSE {test_metrics['rmse']:.1f}  MAE {test_metrics['mae']:.1f}  "
        f"R2 {test_metrics['r2']:.3f}  WAPE {100 * test_metrics['wape']:.1f} %"
    )
    print(
        f"  reference 'same hour and day type' average: RMSE {profile_metrics['rmse']:.1f}  "
        f"MAE {profile_metrics['mae']:.1f}  R2 {profile_metrics['r2']:.3f}"
    )

    final = common.build_pipeline(best, params[best]).fit(X, y)
    first_day, last_day = data["date"].min().date(), data["date"].max().date()
    metadata = {
        "model_name": best,
        "params": params[best],
        "unit": "bikes per hour",
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fit_rows": int(len(data)),
        "fit_date_range": [str(first_day), str(last_day)],
        "training_ranges": common.training_ranges(X),
        "cv": cv_results[best],
        "holdout_test": {"first_day": str(cutoff.date()), "rows": int(len(X_test)), **test_metrics},
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
                "first_day": str(first_day),
                "last_day": str(last_day),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "test_first_day": str(cutoff.date()),
            },
            "validation": {
                "test_split": "chronological, last 20 % of the calendar days",
                "cv": f"GroupKFold({common.N_SPLITS}) over {common.BLOCK_DAYS}-day blocks, training period only",
            },
            "params_status": params_meta.get("status", "unknown"),
            "cv": cv_results,
            "selected_model": best,
            "selected_params": params[best],
            "test": {
                "selected_model": test_metrics,
                "baseline_hour_daytype_mean": profile_metrics,
                "baseline_training_mean": mean_metrics,
            },
            "final_model": {
                "refit_on": "all cleaned rows (training + test period)",
                "rows": int(len(data)),
                "note": "test metrics above were measured on a model trained on the training period only",
            },
            "versions": common.library_versions(),
        }
    )
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_text(reports_dir / "metrics.json", json.dumps(metrics, indent=2) + "\n")
    write_text(reports_dir / "results.md", results_markdown(metrics))
    print(
        f"Saved {model_out.name} (refit on all {len(data)} rows), {reports_dir.name}/metrics.json and "
        f"{reports_dir.name}/results.md in {time.time() - started:.0f}s."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
