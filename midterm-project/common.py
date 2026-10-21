"""Shared code for the shopper purchase-intent project.

Everything that has to be identical between the notebook, ``train.py`` and the
web service lives in this module: the data schema, cleaning, feature
engineering, the scikit-learn pipeline and the evaluation helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
SEED = 42

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "online_shoppers_intention.csv"
PARAMS_PATH = PROJECT_DIR / "params.json"
MODEL_PATH = PROJECT_DIR / "model.bin"
REPORTS_DIR = PROJECT_DIR / "reports"

# --------------------------------------------------------------------------- #
# Schema of the UCI "Online Shoppers Purchasing Intention" dataset
# --------------------------------------------------------------------------- #
TARGET = "Revenue"

COUNT_COLUMNS = ["Administrative", "Informational", "ProductRelated"]
DURATION_COLUMNS = [
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated_Duration",
]
RATE_COLUMNS = ["BounceRates", "ExitRates", "SpecialDay"]  # all live in [0, 1]
VALUE_COLUMNS = ["PageValues"]
BINARY_COLUMNS = ["Weekend"]
# Integer codes without any numeric meaning -> treated as categories.
CODED_COLUMNS = ["OperatingSystems", "Browser", "Region", "TrafficType"]

NUMERIC_RAW = COUNT_COLUMNS + DURATION_COLUMNS + RATE_COLUMNS + VALUE_COLUMNS + BINARY_COLUMNS
CATEGORICAL = ["Month", "VisitorType"] + CODED_COLUMNS
RAW_FEATURES = NUMERIC_RAW + CATEGORICAL

ENGINEERED = [
    "total_pages",
    "total_duration",
    "avg_seconds_per_page",
    "product_page_share",
    "has_page_value",
]
NUMERIC = NUMERIC_RAW + ENGINEERED

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]

MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]


# --------------------------------------------------------------------------- #
# Cleaning and feature engineering
# --------------------------------------------------------------------------- #
def normalize_month(value) -> str:
    """'June' -> 'Jun', 'feb' -> 'Feb' (the raw file spells June out in full)."""
    return str(value).strip()[:3].title()


def to_bool(value) -> bool:
    """Parse booleans that may arrive as bool, 0/1 or 'TRUE'/'false' strings."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the quirks found during EDA and drop duplicated sessions."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["Month"] = df["Month"].map(normalize_month)
    df["Weekend"] = df["Weekend"].map(to_bool).astype(int)
    df[TARGET] = df[TARGET].map(to_bool).astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python download_data.py` first "
            "(see the README for a manual download link)."
        )
    return clean_dataset(pd.read_csv(path))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model input: cleaned raw features + a few derived ones.

    This is the first step of the scikit-learn pipeline, so the web service can
    feed raw session features straight into the saved model.
    """
    out = df[RAW_FEATURES].copy()
    out["Month"] = out["Month"].map(normalize_month)
    out["Weekend"] = out["Weekend"].map(to_bool).astype(int)
    for col in CODED_COLUMNS:
        out[col] = out[col].astype(int).astype(str)

    pages = out[COUNT_COLUMNS].sum(axis=1)
    duration = out[DURATION_COLUMNS].sum(axis=1)
    nonzero_pages = pages.where(pages > 0)  # NaN when a session has no page views

    out["total_pages"] = pages
    out["total_duration"] = duration
    out["avg_seconds_per_page"] = (duration / nonzero_pages).fillna(0.0)
    out["product_page_share"] = (out["ProductRelated"] / nonzero_pages).fillna(0.0)
    out["has_page_value"] = (out["PageValues"] > 0).astype(int)
    return out


# --------------------------------------------------------------------------- #
# Models and pipelines
# --------------------------------------------------------------------------- #
def make_model(name: str, params: dict | None = None, seed: int = SEED):
    """Create an (un-fitted) classifier. ``params`` override the library defaults."""
    params = dict(params or {})
    if name == "logistic_regression":
        return LogisticRegression(**{"max_iter": 2000, "random_state": seed, **params})
    if name == "decision_tree":
        return DecisionTreeClassifier(**{"random_state": seed, **params})
    if name == "random_forest":
        return RandomForestClassifier(**{"random_state": seed, "n_jobs": 1, **params})
    if name == "xgboost":
        return XGBClassifier(
            **{
                "eval_metric": "logloss",
                "tree_method": "hist",
                "random_state": seed,
                "n_jobs": 1,
                **params,
            }
        )
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")


def build_pipeline(name: str, params: dict | None = None, seed: int = SEED) -> Pipeline:
    """feature engineering -> encoding / scaling -> classifier.

    Only the linear model gets log-transformed + standardised numeric features;
    tree-based models are insensitive to monotonic transformations.
    Output of the preprocessing step is always dense: XGBoost treats the
    missing entries of a sparse matrix as NaN, which would make training and
    single-row inference behave differently.
    """
    if name == "logistic_regression":
        numeric = make_pipeline(
            FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
            StandardScaler(),
        )
    else:
        numeric = "passthrough"

    preprocess = ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(engineer)),
            ("prep", preprocess),
            ("model", make_model(name, params, seed)),
        ]
    )


def load_params(path: str | Path = PARAMS_PATH) -> dict:
    """Hyper-parameters per model (written by the notebook, read by train.py)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def cv_scores(estimator, X, y, folds: int = 5, seed: int = SEED, n_jobs: int = -1) -> dict:
    """Stratified k-fold ROC-AUC and PR-AUC (mean and std over folds)."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    res = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        n_jobs=n_jobs,
    )
    return {
        "roc_auc_mean": float(res["test_roc_auc"].mean()),
        "roc_auc_std": float(res["test_roc_auc"].std()),
        "pr_auc_mean": float(res["test_pr_auc"].mean()),
        "pr_auc_std": float(res["test_pr_auc"].std()),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Probability threshold that maximises F1 (used instead of a fixed 0.5)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y_true, proba, threshold: float) -> dict:
    """Ranking metrics plus precision / recall / F1 at a given threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def to_builtin(obj):
    """Recursively convert numpy scalars so the object can be dumped as JSON."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
