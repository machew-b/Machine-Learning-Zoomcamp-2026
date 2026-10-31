"""Shared code for the hospital-readmission project (Diabetes 130-US Hospitals).

Single source of truth used by the notebook, ``train.py``, ``predict.py`` and the tests:

* dataset schema, loading and cleaning,
* the patient-aware validation scheme (hold-out split and cross-validation),
* feature engineering (the first step of the saved scikit-learn Pipeline),
* the model factory / pipeline builder, the metric helpers and the decision rule,
* strict validation of API input records,
* saving and loading of the model bundle.

Keeping all of this in one module guarantees that training and serving cannot drift apart.
"""
from __future__ import annotations

import json
import math
import pickle
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5            # cross-validation folds
TEST_FOLDS = 5          # the hold-out test set is one fold of a 5-fold grouped split (about 20 %)

DATASET_NAME = "Diabetes 130-US Hospitals for Years 1999-2008 (UCI Machine Learning Repository, id 296)"
DATASET_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
CSV_NAME = "diabetic_data.csv"
CSV_ENCODING = "latin-1"      # the file is plain ASCII; latin-1 can never fail to decode
EXPECTED_ROWS = 101766
EXPECTED_PATIENTS = 71518

# The target has three classes. Index = class id used everywhere (probability columns, weights).
CLASS_LABELS = ["NO", ">30", "<30"]
CLASS_KEYS = ["no_readmission", "readmission_after_30_days", "readmission_within_30_days"]
CLASS_DESCRIPTIONS = [
    "not readmitted",
    "readmitted after more than 30 days",
    "readmitted within 30 days",
]
N_CLASSES = 3
EARLY = 2                     # class id of "<30", the clinically most important class
TARGET = "target"             # integer class id added by clean_with_report()
GROUP_COLUMN = "patient_nbr"  # encounters of one patient must never be split between train and test

# --- raw file schema (names after replacing '-' by '_') -------------------------------
MED_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide_metformin",
    "glipizide_metformin", "glimepiride_pioglitazone", "metformin_rosiglitazone", "metformin_pioglitazone",
]
RAW_COLUMNS = (
    [
        "encounter_id", "patient_nbr", "race", "gender", "age", "weight", "admission_type_id",
        "discharge_disposition_id", "admission_source_id", "time_in_hospital", "payer_code",
        "medical_specialty", "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient", "diag_1", "diag_2", "diag_3",
        "number_diagnoses", "max_glu_serum", "A1Cresult",
    ]
    + MED_COLUMNS
    + ["change", "diabetesMed", "readmitted"]
)
assert len(RAW_COLUMNS) == 50
INT_RAW_COLUMNS = [
    "encounter_id", "patient_nbr", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "number_diagnoses",
]

# --- columns the model uses (the Pipeline's first step consumes exactly these) ---------
COUNT_INPUTS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
ID_INPUTS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
TEXT_INPUTS = ["age", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3",
               "max_glu_serum", "A1Cresult", "change", "diabetesMed"]
INPUT_COLUMNS = COUNT_INPUTS + ID_INPUTS + TEXT_INPUTS + MED_COLUMNS
# Kept in the cleaned table for the fairness audit, but deliberately NOT inputs of the model.
AUDIT_COLUMNS = ["race", "gender"]
# Dropped: 'weight' is missing for about 97 % of the encounters; the two identifiers carry no signal.

# --- documented value sets -------------------------------------------------------------
AGE_LABELS = [f"[{start}-{start + 10})" for start in range(0, 100, 10)]
AGE_MID = {label: float(start + 5) for label, start in zip(AGE_LABELS, range(0, 100, 10))}
GLU_VALUES = ["None", "Norm", ">200", ">300"]
A1C_VALUES = ["None", "Norm", ">7", ">8"]
MED_STATUS = ["No", "Steady", "Up", "Down"]
CHANGE_VALUES = ["Ch", "No"]
DIABETES_MED_VALUES = ["Yes", "No"]
# IDS_mapping.csv: 11 'Expired', 13/14 'Hospice', 19/20/21 'Expired ... hospice'. Those patients cannot be readmitted.
EXCLUDED_DISCHARGE_IDS = (11, 13, 14, 19, 20, 21)
# IDS_mapping.csv: 'Not Available', 'NULL', 'Not Mapped', 'Unknown/Invalid' are merged into one level 'unknown'.
UNKNOWN_ADMISSION_TYPE_IDS = (5, 6, 8)
UNKNOWN_ADMISSION_SOURCE_IDS = (9, 15, 17, 20, 21)
UNKNOWN_DISCHARGE_IDS = (18, 25, 26)

# Output columns of build_features().
NUMERIC_FEATURES = COUNT_INPUTS + ["age_mid", "n_prior_visits", "n_meds_active", "n_meds_changed"]
LOG_FEATURES = ["number_outpatient", "number_emergency", "number_inpatient", "n_prior_visits"]  # right-skewed counts
CATEGORICAL_FEATURES = [
    "admission_type", "discharge_disposition", "admission_source", "payer_code", "medical_specialty",
    "diag_1_group", "diag_2_group", "diag_3_group", "max_glu_serum", "A1Cresult", "insulin", "metformin",
    "change", "diabetesMed",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MIN_CATEGORY_COUNT = 100  # categories seen fewer times in training are pooled into one 'infrequent' level

MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest", "hist_gradient_boosting"]
LINEAR_MODELS = {"logistic_regression"}

# Hyper-parameters used until the notebook writes tuned ones to params.json.
DEFAULT_PARAMS = {
    "logistic_regression": {"C": 0.1},
    "decision_tree": {"max_depth": 6, "min_samples_leaf": 100},
    "random_forest": {
        "n_estimators": 200, "max_depth": 14, "min_samples_leaf": 30, "max_features": "sqrt", "max_samples": 0.5,
    },
    "hist_gradient_boosting": {
        "learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 15, "min_samples_leaf": 100,
        "l2_regularization": 1.0,
    },
}

# --- specification of one prediction request (validation, API docs, tests) -------------
# Ranges are generous physical limits, not the ranges seen in training; requests that lie outside
# the training range are accepted but flagged with a warning.
_ICD9_PATTERN = re.compile(r"^(\d{3}(\.\d{1,2})?|V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?)$")
_PAYER_PATTERN = re.compile(r"^[A-Z]{2}$")
_SPECIALTY_PATTERN = re.compile(r"^[A-Za-z0-9 /&,()'\-]{1,60}$")
INPUT_SPEC = {
    "age": {"kind": "choice", "values": AGE_LABELS, "doc": "age group of the patient, one of the ten dataset labels"},
    "admission_type_id": {"kind": "int", "min": 1, "max": 8, "doc": "1 emergency, 2 urgent, 3 elective, 4 newborn, 7 trauma center (5, 6, 8 = unknown)"},
    "discharge_disposition_id": {"kind": "int", "min": 1, "max": 30, "doc": "where the patient goes next, e.g. 1 home, 3 skilled nursing facility, 6 home health service (expired / hospice ids are out of scope)"},
    "admission_source_id": {"kind": "int", "min": 1, "max": 26, "doc": "e.g. 1 physician referral, 4 transfer from a hospital, 7 emergency room"},
    "time_in_hospital": {"kind": "int", "min": 1, "max": 60, "unit": "days", "doc": "length of stay in days (1 to 14 in the training data)"},
    "payer_code": {"kind": "code", "optional": True, "doc": "two-letter payer code such as MC (Medicare), MD (Medicaid), SP (self-pay); omit if unknown"},
    "medical_specialty": {"kind": "text", "optional": True, "doc": "specialty of the admitting physician as written in the dataset, e.g. InternalMedicine; omit if unknown"},
    "num_lab_procedures": {"kind": "int", "min": 0, "max": 300, "doc": "number of lab tests during the stay"},
    "num_procedures": {"kind": "int", "min": 0, "max": 30, "doc": "number of procedures other than lab tests"},
    "num_medications": {"kind": "int", "min": 0, "max": 150, "doc": "number of distinct medications administered"},
    "number_outpatient": {"kind": "int", "min": 0, "max": 500, "doc": "outpatient visits in the year before the stay"},
    "number_emergency": {"kind": "int", "min": 0, "max": 500, "doc": "emergency visits in the year before the stay"},
    "number_inpatient": {"kind": "int", "min": 0, "max": 500, "doc": "inpatient visits in the year before the stay"},
    "diag_1": {"kind": "icd9", "doc": "primary diagnosis, ICD-9 code such as 250.83, 428, V57 or E909"},
    "diag_2": {"kind": "icd9", "optional": True, "doc": "secondary diagnosis (ICD-9); omit if none"},
    "diag_3": {"kind": "icd9", "optional": True, "doc": "additional diagnosis (ICD-9); omit if none"},
    "number_diagnoses": {"kind": "int", "min": 1, "max": 50, "doc": "number of diagnoses entered for the stay"},
    "max_glu_serum": {"kind": "choice", "values": GLU_VALUES, "optional": True, "default": "None", "doc": "glucose serum test result; 'None' = not measured (default)"},
    "A1Cresult": {"kind": "choice", "values": A1C_VALUES, "optional": True, "default": "None", "doc": "HbA1c test result; 'None' = not measured (default)"},
    "change": {"kind": "choice", "values": CHANGE_VALUES, "doc": "'Ch' if the diabetes medication was changed during the stay, else 'No'"},
    "diabetesMed": {"kind": "choice", "values": DIABETES_MED_VALUES, "doc": "'Yes' if any diabetes medication was prescribed"},
    "medications": {"kind": "meds", "optional": True, "doc": "object {drug: No|Steady|Up|Down}; omitted drugs count as 'No'. Drugs: " + ", ".join(MED_COLUMNS)},
}
MAX_BATCH_SIZE = 100


# --------------------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------------------
def normalise_header(name) -> str:
    """'glyburide-metformin' -> 'glyburide_metformin'; also tolerates a BOM or odd casing."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


_COLUMN_ALIASES = {normalise_header(column): column for column in RAW_COLUMNS}


def _canonical_values(series: pd.Series, values) -> pd.Series:
    """Map values that differ only in case or spacing to the documented spelling; leave others alone."""
    lookup = {value.lower(): value for value in values}
    mapped = series.str.lower().map(lookup)
    return mapped.where(mapped.notna(), series)


def standardise_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, check the schema, convert '?' to missing and normalise the value spellings."""
    renamed = {}
    for column in df.columns:
        key = normalise_header(column)
        if key in _COLUMN_ALIASES:
            renamed[column] = _COLUMN_ALIASES[key]
    out = df.rename(columns=renamed)
    missing = [c for c in RAW_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            "The file does not look like the diabetes readmission data. Missing column(s): "
            + ", ".join(missing)
            + ". Columns found: "
            + ", ".join(str(c) for c in df.columns)
        )
    out = out[RAW_COLUMNS].copy()

    for column in RAW_COLUMNS:
        if column in INT_RAW_COLUMNS:
            out[column] = pd.to_numeric(out[column], errors="coerce")
            continue
        series = out[column].astype(object)
        series = series.where(series.isna(), series.astype(str).str.strip())
        # '?' (and an empty cell) is the file's marker for "not recorded"
        series = series.where(~series.isin(["?", ""]), np.nan)
        out[column] = series

    for column in MED_COLUMNS:
        out[column] = _canonical_values(out[column], MED_STATUS)
    out["max_glu_serum"] = _canonical_values(out["max_glu_serum"], GLU_VALUES)
    out["A1Cresult"] = _canonical_values(out["A1Cresult"], A1C_VALUES)
    out["change"] = _canonical_values(out["change"], CHANGE_VALUES)
    out["diabetesMed"] = _canonical_values(out["diabetesMed"], DIABETES_MED_VALUES)

    label_lookup = {label.lower(): label for label in CLASS_LABELS}
    label = out["readmitted"].astype(object).str.lower().map(label_lookup)
    if label.isna().any():
        bad = sorted(set(out.loc[label.isna(), "readmitted"].astype(str)))[:5]
        raise ValueError(f"Unexpected values in column 'readmitted': {bad}; expected {CLASS_LABELS}")
    out["readmitted"] = label
    return out


def load_raw(path) -> pd.DataFrame:
    """Read diabetic_data.csv. Everything is read as text first, so that values such as 'None'
    (test not performed) and 'NO' (not readmitted) are never turned into missing values by pandas."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=CSV_ENCODING)
    return standardise_raw(frame)


def clean_with_report(raw: pd.DataFrame):
    """Apply the cleaning rules and return ``(clean_frame, report)``.

    1. exact duplicate rows and repeated encounter ids are dropped;
    2. encounters that ended in death or hospice care are dropped: those patients cannot be readmitted;
    3. rows without a usable patient id, target or count are dropped;
    4. the integer class id is added as column ``target``.

    Several encounters of the same patient are kept (they are real events); the validation
    scheme keeps all encounters of a patient on the same side of every split instead.
    """
    report = {"rows_raw": int(len(raw))}
    df = raw.drop_duplicates()
    report["duplicate_rows_dropped"] = int(len(raw) - len(df))

    repeated = df["encounter_id"].duplicated(keep="first")
    report["duplicate_encounter_ids_dropped"] = int(repeated.sum())
    df = df[~repeated]

    excluded = df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)
    report["expired_or_hospice_dropped"] = int(excluded.sum())
    df = df[~excluded]

    before = len(df)
    df = df.dropna(subset=[GROUP_COLUMN, "readmitted"] + COUNT_INPUTS + ID_INPUTS)
    report["rows_dropped_missing_key_fields"] = int(before - len(df))
    if len(df) == 0:
        raise ValueError("No rows left after cleaning.")

    df = df.copy()
    for column in INT_RAW_COLUMNS:
        df[column] = df[column].astype(np.int64)
    df[TARGET] = df["readmitted"].map({label: i for i, label in enumerate(CLASS_LABELS)}).astype(int)
    df = df.reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    report["patients_clean"] = int(df[GROUP_COLUMN].nunique())
    report["class_counts"] = {label: int((df[TARGET] == i).sum()) for i, label in enumerate(CLASS_LABELS)}
    return df, report


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    return clean_with_report(raw)[0]


# --------------------------------------------------------------------------------------
# Validation scheme: patient-aware hold-out split and cross-validation
# --------------------------------------------------------------------------------------
def grouped_stratified_split(y, groups, n_splits: int = TEST_FOLDS, seed: int = RANDOM_STATE):
    """Boolean masks (train, test). The test set is one fold of a stratified grouped k-fold.

    All encounters of a patient land on the same side (a patient's later stays resemble the earlier ones,
    so a random row split would flatter the model), and the class proportions stay close to the overall ones.
    """
    y = np.asarray(y)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, np.asarray(groups)))
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_idx] = True
    return ~test_mask, test_mask


def grouped_cv_splitter(n_splits: int = N_SPLITS, seed: int = RANDOM_STATE) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def sample_rows_by_group(groups, n_rows: int, seed: int = RANDOM_STATE) -> np.ndarray:
    """Boolean mask of about ``n_rows`` rows made of whole groups (used to keep hyper-parameter searches fast)."""
    groups = np.asarray(groups)
    if n_rows >= len(groups):
        return np.ones(len(groups), dtype=bool)
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(unique))
    cumulative = np.cumsum(counts[order])
    keep = int(np.searchsorted(cumulative, n_rows)) + 1
    chosen = np.zeros(len(unique), dtype=bool)
    chosen[order[:keep]] = True
    return chosen[inverse.reshape(-1)]


# --------------------------------------------------------------------------------------
# Feature engineering (first step of the Pipeline, so training and serving share it)
# --------------------------------------------------------------------------------------
def icd9_group(code) -> str:
    """Group an ICD-9 diagnosis code into the nine classes of Strack et al. (2014), plus 'Unknown'."""
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return "Unknown"
    text = str(code).strip().upper()
    if text in ("", "?", "NAN", "NONE"):
        return "Unknown"
    if text[0] in "VE":                      # supplementary classification codes
        return "Other"
    try:
        whole = int(float(text))
    except ValueError:
        return "Other"
    if whole == 250:
        return "Diabetes"
    if 390 <= whole <= 459 or whole == 785:
        return "Circulatory"
    if 460 <= whole <= 519 or whole == 786:
        return "Respiratory"
    if 520 <= whole <= 579 or whole == 787:
        return "Digestive"
    if 800 <= whole <= 999:
        return "Injury"
    if 710 <= whole <= 739:
        return "Musculoskeletal"
    if 580 <= whole <= 629 or whole == 788:
        return "Genitourinary"
    if 140 <= whole <= 239:
        return "Neoplasms"
    return "Other"


def _icd_groups(series: pd.Series) -> pd.Series:
    series = series.astype(object)
    lookup = {value: icd9_group(value) for value in series.dropna().unique()}
    return series.map(lookup).astype(object).where(series.notna(), "Unknown")


def _clean_text(series: pd.Series, fill: str) -> pd.Series:
    series = series.astype(object)
    series = series.where(series.notna(), fill)
    series = series.astype(str).str.strip()
    return series.where(series != "", fill).astype(object)


def _id_category(series: pd.Series, unknown_ids) -> pd.Series:
    numbers = pd.to_numeric(series, errors="coerce")
    out = pd.Series("unknown", index=series.index, dtype=object)
    valid = numbers.notna() & ~numbers.isin(unknown_ids)
    out[valid] = numbers[valid].astype(int).astype(str)
    return out


def build_features(X: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw input columns into model features (race and gender are not among them)."""
    missing = [c for c in INPUT_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError("Missing input column(s): " + ", ".join(missing))

    out = pd.DataFrame(index=X.index)
    for column in COUNT_INPUTS:
        out[column] = pd.to_numeric(X[column], errors="coerce").astype(float)
    out["age_mid"] = X["age"].astype(object).map(AGE_MID).astype(float)   # decade mid-point, NaN if unknown
    out["n_prior_visits"] = (
        out["number_outpatient"].fillna(0) + out["number_emergency"].fillna(0) + out["number_inpatient"].fillna(0)
    )

    meds = X[MED_COLUMNS].astype(object)
    out["n_meds_active"] = (meds.notna() & (meds != "No")).sum(axis=1).astype(float)
    out["n_meds_changed"] = meds.isin(["Up", "Down"]).sum(axis=1).astype(float)

    out["admission_type"] = _id_category(X["admission_type_id"], UNKNOWN_ADMISSION_TYPE_IDS)
    out["discharge_disposition"] = _id_category(X["discharge_disposition_id"], UNKNOWN_DISCHARGE_IDS)
    out["admission_source"] = _id_category(X["admission_source_id"], UNKNOWN_ADMISSION_SOURCE_IDS)
    out["payer_code"] = _clean_text(X["payer_code"], "Unknown")
    out["medical_specialty"] = _clean_text(X["medical_specialty"], "Unknown")
    for number in (1, 2, 3):
        out[f"diag_{number}_group"] = _icd_groups(X[f"diag_{number}"])
    out["max_glu_serum"] = _clean_text(X["max_glu_serum"], "None")
    out["A1Cresult"] = _clean_text(X["A1Cresult"], "None")
    out["insulin"] = _clean_text(X["insulin"], "No")
    out["metformin"] = _clean_text(X["metformin"], "No")
    out["change"] = _clean_text(X["change"], "No")
    out["diabetesMed"] = _clean_text(X["diabetesMed"], "No")
    return out[FEATURE_COLUMNS]


# --------------------------------------------------------------------------------------
# Models and pipelines
# --------------------------------------------------------------------------------------
def _make_estimator(name: str, p: dict, n_jobs: int = 1):
    if name == "logistic_regression":
        return LogisticRegression(C=p["C"], max_iter=1000, random_state=RANDOM_STATE)
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], random_state=RANDOM_STATE
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"],
            max_samples=p.get("max_samples"),
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=p["learning_rate"],
            max_iter=p["max_iter"],
            max_leaf_nodes=p["max_leaf_nodes"],
            min_samples_leaf=p["min_samples_leaf"],
            l2_regularization=p["l2_regularization"],
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model '{name}'. Choose one of: {', '.join(MODEL_NAMES)}")


def _one_hot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_COUNT, sparse_output=False)


def _linear_preprocessor() -> ColumnTransformer:
    plain_numeric = [c for c in NUMERIC_FEATURES if c not in LOG_FEATURES]
    return ColumnTransformer(
        transformers=[
            ("categories", _one_hot(), CATEGORICAL_FEATURES),
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                plain_numeric,
            ),
            (
                "skewed",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]
                ),
                LOG_FEATURES,
            ),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


def _tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categories", _one_hot(), CATEGORICAL_FEATURES),
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


def build_pipeline(model_name: str, params: dict | None = None, n_jobs: int = 1) -> Pipeline:
    """Unfitted Pipeline: feature engineering -> preprocessing -> classifier.

    ``n_jobs`` only affects the speed of the random forest (the result does not depend on it).
    """
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {', '.join(MODEL_NAMES)}")
    merged = {**DEFAULT_PARAMS[model_name], **(params or {})}
    preprocess = _linear_preprocessor() if model_name in LINEAR_MODELS else _tree_preprocessor()
    return Pipeline(
        [
            ("features", FunctionTransformer(build_features, validate=False)),
            ("preprocess", preprocess),
            ("model", _make_estimator(model_name, merged, n_jobs=n_jobs)),
        ]
    )


def make_serving_ready(pipeline: Pipeline) -> Pipeline:
    """Single-threaded prediction: the service answers one small request at a time."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=1)
    return pipeline


def plain_params(search_params: dict) -> dict:
    """RandomizedSearchCV names ('model__C') -> plain names ('C')."""
    return {key.split("__")[-1]: to_builtin(value) for key, value in search_params.items()}


def to_builtin(value):
    """Convert numpy scalars/arrays to plain Python so that they can be written as JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


# --------------------------------------------------------------------------------------
# Metrics, decision rule and cross-validation
# --------------------------------------------------------------------------------------
def predict_proba(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Class probabilities with columns in the order of CLASS_LABELS."""
    proba = np.asarray(pipeline.predict_proba(X), dtype=float)
    classes = [int(c) for c in pipeline.classes_]
    if classes != list(range(N_CLASSES)):
        full = np.zeros((len(proba), N_CLASSES))
        full[:, classes] = proba
        proba = full
    return proba


def decide(proba, weights) -> np.ndarray:
    """Decision rule: the class with the largest weight x probability."""
    return np.argmax(np.asarray(proba, dtype=float) * np.asarray(weights, dtype=float), axis=1)


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.bincount(y_true * N_CLASSES + y_pred, minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)


def _macro_f1_from_confusion(cm: np.ndarray) -> float:
    true_pos = np.diag(cm).astype(float)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * true_pos, denominator, out=np.zeros(N_CLASSES), where=denominator > 0)
    return float(f1.mean())


# Search grid of the decision weights (the weight of class "NO" is fixed to 1).
WEIGHT_GRID_30 = np.round(np.arange(0.5, 3.0001, 0.1), 2)     # class ">30"
WEIGHT_GRID_EARLY = np.round(np.arange(1.0, 12.0001, 0.25), 2)  # class "<30"


def tune_decision_weights(y_true, proba):
    """Class weights that maximise the macro-F1 of ``decide`` on the given (out-of-fold) predictions.

    The classifier is trained on the natural class frequencies, so its probabilities are honest but the
    rare class "<30" is almost never the arg-max. Re-weighting the probabilities moves the operating point.
    Returns ``(weights, macro_f1)``.
    """
    y = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    best_weights, best_f1 = [1.0, 1.0, 1.0], -1.0
    for w_late in WEIGHT_GRID_30:
        for w_early in WEIGHT_GRID_EARLY:
            weights = np.array([1.0, w_late, w_early])
            f1 = _macro_f1_from_confusion(_confusion(y, np.argmax(proba * weights, axis=1)))
            if f1 > best_f1 + 1e-12:
                best_weights, best_f1 = [1.0, float(w_late), float(w_early)], f1
    return best_weights, float(best_f1)


def _auc(y_binary, score) -> float:
    y_binary = np.asarray(y_binary, dtype=int)
    if y_binary.min() == y_binary.max():
        return float("nan")
    return float(roc_auc_score(y_binary, score))


def probability_metrics(y_true, proba) -> dict:
    """Threshold-free metrics of the predicted probabilities."""
    y = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    aucs = [_auc(y == k, proba[:, k]) for k in range(N_CLASSES)]
    early = (y == EARLY).astype(int)
    return {
        "log_loss": float(log_loss(y, np.clip(proba, 1e-15, 1.0), labels=list(range(N_CLASSES)))),
        "roc_auc_macro_ovr": float(np.nanmean(aucs)),
        "roc_auc_early": aucs[EARLY],
        "roc_auc_any_readmission": _auc(y != 0, 1.0 - proba[:, 0]),
        "pr_auc_early": float(average_precision_score(early, proba[:, EARLY])) if early.max() > 0 else float("nan"),
    }


def decision_metrics(y_true, proba, weights) -> dict:
    """Metrics of the decisions taken with the weighted arg-max rule."""
    y = np.asarray(y_true, dtype=int)
    pred = decide(proba, weights)
    cm = _confusion(y, pred)
    true_pos = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(true_pos, predicted, out=np.zeros(N_CLASSES), where=predicted > 0)
    recall = np.divide(true_pos, support, out=np.zeros(N_CLASSES), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(N_CLASSES), where=(precision + recall) > 0)
    return {
        "weights": [float(w) for w in weights],
        "accuracy": float(true_pos.sum() / max(1, cm.sum())),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "predicted": int(predicted[i]),
            }
            for i, label in enumerate(CLASS_LABELS)
        },
        "confusion_matrix": cm.tolist(),   # rows = true class, columns = predicted class (order of CLASS_LABELS)
    }


def classification_metrics(y_true, proba, weights=None) -> dict:
    metrics = probability_metrics(y_true, proba)
    if weights is not None:
        metrics.update(decision_metrics(y_true, proba, weights))
    return metrics


def prior_probabilities(y_train, n_rows: int) -> np.ndarray:
    """Naive reference model: every row gets the class frequencies of the training data."""
    y = np.asarray(y_train, dtype=int)
    frequencies = np.bincount(y, minlength=N_CLASSES) / len(y)
    return np.tile(frequencies, (n_rows, 1))


FOLD_METRICS = ("log_loss", "roc_auc_macro_ovr", "roc_auc_early", "pr_auc_early")


def summarise_folds(folds: list) -> dict:
    """Mean and standard deviation of every threshold-free metric over the cross-validation folds."""
    summary = {}
    for metric in FOLD_METRICS:
        values = np.array([f[metric] for f in folds], dtype=float)
        summary[f"{metric}_mean"] = float(np.nanmean(values))
        summary[f"{metric}_std"] = float(np.nanstd(values))
    return summary


def cross_validate_model(pipeline, X: pd.DataFrame, y: pd.Series, groups, n_splits: int = N_SPLITS, splitter=None) -> dict:
    """Patient-aware cross-validation; returns per-fold metrics, mean/std and out-of-fold probabilities.

    ``splitter`` can be replaced (for example by a shuffled StratifiedKFold) to demonstrate leakage.
    """
    splitter = splitter if splitter is not None else grouped_cv_splitter(n_splits)
    oof = np.full((len(y), N_CLASSES), np.nan)
    folds = []
    for train_idx, valid_idx in splitter.split(X, y, groups):
        model = clone(pipeline)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = predict_proba(model, X.iloc[valid_idx])
        oof[valid_idx] = proba
        folds.append(probability_metrics(y.iloc[valid_idx], proba))
    return {"folds": folds, "summary": summarise_folds(folds), "oof_proba": oof}


def training_ranges(X: pd.DataFrame) -> dict:
    """Minimum and maximum of every count input as seen in training."""
    ranges = {}
    for column in COUNT_INPUTS:
        values = pd.to_numeric(X[column], errors="coerce").dropna()
        ranges[column] = [int(values.min()), int(values.max())]
    return ranges


def training_categories(X: pd.DataFrame) -> dict:
    """Payer codes and medical specialties seen in training (used to warn about unseen values)."""
    features = build_features(X)
    return {
        column: sorted(v for v in features[column].unique() if v != "Unknown")
        for column in ("payer_code", "medical_specialty")
    }


# Groups of raw inputs permuted together when measuring the importance of a variable.
PERMUTATION_GROUPS = {
    **{column: [column] for column in COUNT_INPUTS + ID_INPUTS + TEXT_INPUTS if column not in ("diag_1", "diag_2", "diag_3")},
    "diag_1": ["diag_1"],
    "diag_2 and diag_3": ["diag_2", "diag_3"],
    "all 23 diabetes medications": list(MED_COLUMNS),
    "insulin": ["insulin"],
}


def grouped_permutation_importance(pipeline, X: pd.DataFrame, y, groups: dict | None = None, n_repeats: int = 3,
                                   seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Permutation importance on the raw model inputs.

    Every group of columns is shuffled jointly (rows permuted together) and the increase of the
    log-loss and the drop of the ROC-AUC for "<30" are recorded; larger means more important.
    """
    groups = groups or PERMUTATION_GROUPS
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    base = probability_metrics(y, predict_proba(pipeline, X))
    rows = []
    for name, columns in groups.items():
        loss_increase, auc_drop = [], []
        for _ in range(n_repeats):
            order = rng.permutation(len(X))
            shuffled = X[columns].iloc[order]
            shuffled.index = X.index
            permuted = X.copy()
            permuted[columns] = shuffled
            metrics = probability_metrics(y, predict_proba(pipeline, permuted))
            loss_increase.append(metrics["log_loss"] - base["log_loss"])
            auc_drop.append(base["roc_auc_early"] - metrics["roc_auc_early"])
        rows.append(
            {
                "variable": name,
                "log_loss_increase": float(np.mean(loss_increase)),
                "log_loss_increase_std": float(np.std(loss_increase)),
                "auc_early_drop": float(np.mean(auc_drop)),
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss_increase", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Parameters file, model bundle, versions
# --------------------------------------------------------------------------------------
def load_params(path="params.json"):
    """Return ``(params_per_model, meta)``; falls back to DEFAULT_PARAMS if the file is missing."""
    path = Path(path)
    if not path.exists():
        return {k: dict(v) for k, v in DEFAULT_PARAMS.items()}, {"status": "defaults", "note": f"{path.name} not found"}
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.pop("_meta", {})
    unknown = sorted(set(data) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"{path.name} contains unknown model(s): {', '.join(unknown)}")
    params = {name: {**DEFAULT_PARAMS[name], **data.get(name, {})} for name in MODEL_NAMES}
    return params, meta


def save_params(path, params: dict, meta: dict) -> None:
    payload = {"_meta": to_builtin(meta)}
    payload.update({name: to_builtin(params[name]) for name in MODEL_NAMES})
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def library_versions() -> dict:
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }


def save_bundle(path, pipeline, metadata: dict) -> None:
    with open(path, "wb") as handle:
        pickle.dump({"pipeline": pipeline, "metadata": metadata}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_bundle(path) -> dict:
    """Load a bundle written by train.py. Only load files you created yourself (pickle)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file '{path.name}' not found. Run `python train.py` first "
            "(the Docker build does this for you)."
        )
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "metadata" not in bundle:
        raise ValueError(f"'{path.name}' is not a model bundle written by train.py.")
    return bundle


# --------------------------------------------------------------------------------------
# API input validation
# --------------------------------------------------------------------------------------
def _is_real(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_record(record):
    """Validate one request record. Returns ``(clean_record, errors)``; ``clean_record`` is None on error.

    The clean record has every field of INPUT_SPEC; optional fields that were omitted (or null) hold their
    default (None, or "None" for the two lab results) and ``medications`` is complete (all 23 drugs).
    """
    if not isinstance(record, dict):
        return None, ["a record must be a JSON object"]
    errors = []
    unknown = sorted(set(record) - set(INPUT_SPEC))
    if unknown:
        errors.append("unknown field(s): " + ", ".join(unknown))
    missing = [name for name, spec in INPUT_SPEC.items() if not spec.get("optional") and record.get(name) is None]
    if missing:
        errors.append("missing field(s): " + ", ".join(missing))

    clean = {}
    for name, spec in INPUT_SPEC.items():
        value = record.get(name)
        kind = spec["kind"]
        if value is None:
            if spec.get("optional"):
                clean[name] = {drug: "No" for drug in MED_COLUMNS} if kind == "meds" else spec.get("default")
            continue
        if kind == "int":
            if not _is_real(value) or float(value) != int(value):
                errors.append(f"{name} must be an integer")
                continue
            number = int(value)
            if not spec["min"] <= number <= spec["max"]:
                errors.append(f"{name} must be between {spec['min']} and {spec['max']}, got {number}")
                continue
            if name == "discharge_disposition_id" and number in EXCLUDED_DISCHARGE_IDS:
                errors.append(
                    f"discharge_disposition_id {number} (patient expired or went to hospice) is outside the scope "
                    "of the model: those patients cannot be readmitted"
                )
                continue
            clean[name] = number
        elif kind == "choice":
            if not isinstance(value, str) or value not in spec["values"]:
                errors.append(f"{name} must be one of: " + ", ".join(spec["values"]))
                continue
            clean[name] = value
        elif kind == "code":
            if not isinstance(value, str) or not _PAYER_PATTERN.match(value.strip()):
                errors.append(f"{name} must be a two-letter upper-case code such as MC")
                continue
            clean[name] = value.strip()
        elif kind == "text":
            if not isinstance(value, str) or not _SPECIALTY_PATTERN.match(value.strip()):
                errors.append(f"{name} must be text of 1 to 60 characters (letters, digits, space and / & , ( ) ' -)")
                continue
            clean[name] = value.strip()
        elif kind == "icd9":
            code = value.strip().upper() if isinstance(value, str) else None
            if code is None or not _ICD9_PATTERN.match(code):
                errors.append(f"{name} must be an ICD-9 code such as 250.83, 428, V57 or E909, got {value!r}")
                continue
            clean[name] = code
        elif kind == "meds":
            if not isinstance(value, dict):
                errors.append("medications must be an object such as {\"insulin\": \"Up\"}")
                continue
            drugs = {drug: "No" for drug in MED_COLUMNS}
            problems = []
            for drug, status in value.items():
                if drug not in drugs:
                    problems.append(f"unknown drug {drug!r}")
                elif status not in MED_STATUS:
                    problems.append(f"{drug} must be one of: " + ", ".join(MED_STATUS))
                else:
                    drugs[drug] = status
            if problems:
                errors.append("medications: " + "; ".join(problems))
                continue
            clean[name] = drugs
    if errors:
        return None, errors
    return clean, []


def records_to_frame(records: list) -> pd.DataFrame:
    """List of validated records -> DataFrame with the columns the Pipeline expects."""
    rows = []
    for record in records:
        row = {column: record[column] for column in COUNT_INPUTS + ID_INPUTS + TEXT_INPUTS}
        row.update({drug: record["medications"][drug] for drug in MED_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


def input_warnings(record: dict, ranges: dict | None, categories: dict | None = None) -> list:
    """Non-blocking remarks about a validated record."""
    warnings = []
    for column, (low, high) in (ranges or {}).items():
        value = record[column]
        if value < low or value > high:
            warnings.append(
                f"{column}={value} is outside the range seen in training ({low} to {high}); "
                "the prediction is an extrapolation"
            )
    for column, known in (categories or {}).items():
        value = record.get(column)
        if value is not None and known and value not in known:
            warnings.append(f"{column}={value!r} was not seen in training and is treated as an unusual level")
    if record["num_procedures"] > 0 and record["num_medications"] == 0:
        warnings.append("num_medications is 0 although the stay includes procedures; please check the input")
    return warnings
