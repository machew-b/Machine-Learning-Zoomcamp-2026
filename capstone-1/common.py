"""Shared code for the Seoul bike-demand project.

Single source of truth used by the notebook, ``train.py``, ``predict.py`` and the tests:

* dataset schema, loading and cleaning,
* feature engineering (the first step of the saved scikit-learn Pipeline),
* the model factory / pipeline builder and the metric and cross-validation helpers,
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
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 5            # cross-validation folds
BLOCK_DAYS = 7          # consecutive days that always stay in the same CV fold
TEST_FRACTION = 0.2     # the last 20 % of the calendar days form the hold-out test set

DATASET_NAME = "Seoul Bike Sharing Demand (UCI Machine Learning Repository, id 560)"
DATASET_URL = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
CSV_NAME = "SeoulBikeData.csv"
CSV_ENCODING = "latin-1"      # the file stores the degree sign as the single byte 0xB0
DATE_FORMAT = "%d/%m/%Y"      # e.g. 01/12/2017 is 1 December 2017
EXPECTED_ROWS = 8760          # 365 days x 24 hours

TARGET = "rented_bike_count"

WEATHER_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "visibility_10m",
    "dew_point_c",
    "solar_radiation_mj_m2",
    "rainfall_mm",
    "snowfall_cm",
]
# Columns the service needs from a client (and that the Pipeline's first step consumes).
INPUT_COLUMNS = ["date", "hour", "holiday"] + WEATHER_COLUMNS
# Columns of the raw file after renaming, in file order.
RAW_COLUMNS = ["date", TARGET, "hour"] + WEATHER_COLUMNS + ["season", "holiday", "functioning_day"]
NUMERIC_RAW_COLUMNS = [TARGET, "hour"] + WEATHER_COLUMNS

# Output columns of build_features().
BINARY_COLUMNS = ["is_weekend", "holiday", "day_off", "is_raining", "is_snowing"]
FEATURE_COLUMNS = ["hour", "day_of_week", "hour_daytype"] + BINARY_COLUMNS + WEATHER_COLUMNS
SKEWED_COLUMNS = ["rainfall_mm", "snowfall_cm"]   # zero-inflated: log1p before scaling (linear model)

SEASONS = ("Winter", "Spring", "Summer", "Autumn")

# normalised raw header -> canonical column name (a few documented spelling variants included)
_COLUMN_ALIASES = {
    "date": "date",
    "rented_bike_count": TARGET,
    "hour": "hour",
    "temperature_c": "temperature_c",
    "temperature": "temperature_c",
    "humidity": "humidity_pct",
    "wind_speed_m_s": "wind_speed_ms",
    "visibility_10m": "visibility_10m",
    "dew_point_temperature_c": "dew_point_c",
    "solar_radiation_mj_m2": "solar_radiation_mj_m2",
    "rainfall_mm": "rainfall_mm",
    "snowfall_cm": "snowfall_cm",
    "seasons": "season",
    "season": "season",
    "holiday": "holiday",
    "functioning_day": "functioning_day",
    "functional_day": "functioning_day",
}
_HOLIDAY_VALUES = {"holiday": True, "no holiday": False, "no_holiday": False}
_FUNCTIONING_VALUES = {"yes": True, "fun": True, "no": False, "nofunc": False}

MODEL_NAMES = ["ridge", "decision_tree", "random_forest", "hist_gradient_boosting"]
LINEAR_MODELS = {"ridge"}

# Hyper-parameters used until the notebook writes tuned ones to params.json.
DEFAULT_PARAMS = {
    "ridge": {"alpha": 1.0, "log_target": True},
    "decision_tree": {"max_depth": 10, "min_samples_leaf": 10},
    "random_forest": {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 3, "max_features": 0.6},
    "hist_gradient_boosting": {
        "learning_rate": 0.08,
        "max_iter": 400,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
    },
}

# Specification of one prediction request (used for validation, the API docs and the tests).
# The ranges are generous physical limits, not the ranges seen in training; requests that lie
# outside the training range are accepted but flagged with a warning.
INPUT_SPEC = {
    "date": {"kind": "date", "doc": "calendar date, ISO format YYYY-MM-DD"},
    "hour": {"kind": "int", "min": 0, "max": 23, "doc": "hour of the day, 0-23"},
    "holiday": {"kind": "bool", "doc": "true if the date is a public holiday in Seoul"},
    "temperature_c": {"kind": "float", "min": -50.0, "max": 60.0, "unit": "degC", "doc": "air temperature"},
    "humidity_pct": {"kind": "float", "min": 0.0, "max": 100.0, "unit": "%", "doc": "relative humidity"},
    "wind_speed_ms": {"kind": "float", "min": 0.0, "max": 60.0, "unit": "m/s", "doc": "wind speed"},
    "visibility_10m": {"kind": "float", "min": 0.0, "max": 10000.0, "unit": "10 m", "doc": "visibility in units of 10 m"},
    "dew_point_c": {"kind": "float", "min": -60.0, "max": 40.0, "unit": "degC", "doc": "dew point temperature"},
    "solar_radiation_mj_m2": {"kind": "float", "min": 0.0, "max": 6.0, "unit": "MJ/m2", "doc": "solar radiation in the hour"},
    "rainfall_mm": {"kind": "float", "min": 0.0, "max": 300.0, "unit": "mm", "doc": "rainfall in the hour"},
    "snowfall_cm": {"kind": "float", "min": 0.0, "max": 100.0, "unit": "cm", "doc": "snowfall in the hour"},
}
MAX_BATCH_SIZE = 168   # one week of hourly records per batch request


# --------------------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------------------
def normalise_header(name) -> str:
    """'Temperature(°C)' -> 'temperature_c'. Robust to the degree sign being decoded wrongly."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def standardise_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, check the schema, parse dates and normalise the categorical columns."""
    renamed = {}
    for column in df.columns:
        key = normalise_header(column)
        if key in _COLUMN_ALIASES:
            renamed[column] = _COLUMN_ALIASES[key]
    out = df.rename(columns=renamed)
    missing = [c for c in RAW_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            "The file does not look like the Seoul bike-sharing data. Missing column(s): "
            + ", ".join(missing)
            + ". Columns found: "
            + ", ".join(str(c) for c in df.columns)
        )
    out = out[RAW_COLUMNS].copy()

    try:
        out["date"] = pd.to_datetime(out["date"], format=DATE_FORMAT)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Column 'date' is not in the expected {DATE_FORMAT} format: {exc}") from exc

    for column in NUMERIC_RAW_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["holiday"] = _map_flag(out["holiday"], _HOLIDAY_VALUES, "holiday")
    out["functioning_day"] = _map_flag(out["functioning_day"], _FUNCTIONING_VALUES, "functioning_day")

    season_key = out["season"].astype(str).str.strip().str.lower()
    season = season_key.map({s.lower(): s for s in SEASONS})
    if season.isna().any():
        raise ValueError(f"Unexpected values in column 'season': {sorted(set(season_key[season.isna()]))}")
    out["season"] = season.astype(str)
    return out


def _map_flag(series: pd.Series, mapping: dict, column: str) -> pd.Series:
    key = series.astype(str).str.strip().str.lower()
    mapped = key.map(mapping)
    if mapped.isna().any():
        raise ValueError(f"Unexpected values in column '{column}': {sorted(set(key[mapped.isna()]))}")
    return mapped.astype(bool)


def load_raw(path) -> pd.DataFrame:
    """Read SeoulBikeData.csv and return it with canonical column names and dtypes."""
    return standardise_raw(pd.read_csv(path, encoding=CSV_ENCODING))


def clean_with_report(raw: pd.DataFrame):
    """Apply the cleaning rules and return ``(clean_frame, report)``.

    1. exact duplicate rows are dropped;
    2. hours in which the bike system was not operating are dropped: the count is 0 by
       construction there, it is not demand;
    3. rows without a usable date, hour or count are dropped;
    4. the table is sorted chronologically.
    """
    report = {"rows_raw": int(len(raw))}
    df = raw.drop_duplicates()
    report["duplicate_rows_dropped"] = int(len(raw) - len(df))

    closed = ~df["functioning_day"]
    report["non_functioning_rows_dropped"] = int(closed.sum())
    df = df[~closed]

    before = len(df)
    df = df.dropna(subset=["date", "hour", TARGET])
    report["rows_dropped_missing_key_fields"] = int(before - len(df))

    if len(df) == 0:
        raise ValueError("No rows left after cleaning.")
    if not ((df["hour"] >= 0) & (df["hour"] <= 23) & (df["hour"] % 1 == 0)).all():
        raise ValueError("Column 'hour' must contain integers between 0 and 23.")
    if (df[TARGET] < 0).any():
        raise ValueError(f"Column '{TARGET}' contains negative counts.")

    df = df.drop(columns=["functioning_day"]).copy()
    df["hour"] = df["hour"].astype(int)
    df[TARGET] = df[TARGET].astype(float)
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    return df, report


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    return clean_with_report(raw)[0]


# --------------------------------------------------------------------------------------
# Validation scheme: chronological hold-out + blocked cross-validation
# --------------------------------------------------------------------------------------
def time_split_masks(dates: pd.Series, test_fraction: float = TEST_FRACTION):
    """Boolean masks (train, test) and the first test day.

    The test set is the last ``test_fraction`` of the distinct calendar days. A random split
    would put neighbouring hours (almost identical weather and demand) on both sides and
    flatter the model; a chronological split mimics forecasting the future.
    """
    days = pd.DatetimeIndex(pd.to_datetime(dates).dt.normalize().unique()).sort_values()
    n_test = max(1, int(round(len(days) * test_fraction)))
    if n_test >= len(days):
        raise ValueError("Not enough distinct days to hold out a test set.")
    cutoff = days[-n_test]
    is_test = (pd.to_datetime(dates) >= cutoff).to_numpy()
    return ~is_test, is_test, pd.Timestamp(cutoff)


def block_groups(dates: pd.Series, block_days: int = BLOCK_DAYS) -> np.ndarray:
    """Group id per row: consecutive ``block_days``-day blocks (used with GroupKFold)."""
    day = pd.to_datetime(dates).dt.normalize()
    return ((day - day.min()).dt.days // block_days).to_numpy()


# --------------------------------------------------------------------------------------
# Feature engineering (first step of the Pipeline, so training and serving share it)
# --------------------------------------------------------------------------------------
def build_features(X: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw input columns into model features."""
    missing = [c for c in INPUT_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError("Missing input column(s): " + ", ".join(missing))

    dates = pd.to_datetime(X["date"])
    out = pd.DataFrame(index=X.index)
    out["hour"] = X["hour"].astype(int)
    out["day_of_week"] = dates.dt.dayofweek.astype(int)            # Monday = 0 ... Sunday = 6
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["holiday"] = X["holiday"].astype(bool).astype(int)
    out["day_off"] = ((out["is_weekend"] + out["holiday"]) > 0).astype(int)
    # hour combined with working day / day off: weekday commuting peaks differ from weekend leisure
    out["hour_daytype"] = [
        f"{h:02d}_{'off' if d else 'work'}" for h, d in zip(out["hour"].tolist(), out["day_off"].tolist())
    ]
    for column in WEATHER_COLUMNS:
        out[column] = pd.to_numeric(X[column], errors="coerce").astype(float)
    # A relative humidity of exactly 0 % does not occur in Seoul: treat it as a missing reading.
    out["humidity_pct"] = out["humidity_pct"].where(out["humidity_pct"] > 0)
    out["is_raining"] = (out["rainfall_mm"].fillna(0) > 0).astype(int)
    out["is_snowing"] = (out["snowfall_cm"].fillna(0) > 0).astype(int)
    return out[FEATURE_COLUMNS]


# --------------------------------------------------------------------------------------
# Models and pipelines
# --------------------------------------------------------------------------------------
def _make_estimator(name: str, p: dict):
    if name == "ridge":
        # By default the linear model is fitted on log(1 + count): demand effects are multiplicative and the
        # count is right-skewed. log_target=False fits the raw count (the notebook compares both).
        use_log = bool(p.get("log_target", True))
        return TransformedTargetRegressor(
            regressor=Ridge(alpha=p["alpha"]),
            func=np.log1p if use_log else None,
            inverse_func=np.expm1 if use_log else None,
            check_inverse=False,
        )
    if name == "decision_tree":
        return DecisionTreeRegressor(
            max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], random_state=RANDOM_STATE
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            max_features=p["max_features"],
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=p["learning_rate"],
            max_iter=p["max_iter"],
            max_leaf_nodes=p["max_leaf_nodes"],
            min_samples_leaf=p["min_samples_leaf"],
            l2_regularization=p["l2_regularization"],
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model '{name}'. Choose one of: {', '.join(MODEL_NAMES)}")


def _linear_preprocessor() -> ColumnTransformer:
    plain_numeric = [c for c in WEATHER_COLUMNS if c not in SKEWED_COLUMNS]
    return ColumnTransformer(
        transformers=[
            ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["hour_daytype", "day_of_week"]),
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
                SKEWED_COLUMNS,
            ),
            ("flags", "passthrough", BINARY_COLUMNS),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


def _tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), WEATHER_COLUMNS),
            ("time_and_flags", "passthrough", ["hour", "day_of_week"] + BINARY_COLUMNS),
        ],
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


def build_pipeline(model_name: str, params: dict | None = None) -> Pipeline:
    """Unfitted Pipeline: feature engineering -> preprocessing -> regressor."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {', '.join(MODEL_NAMES)}")
    merged = {**DEFAULT_PARAMS[model_name], **(params or {})}
    preprocess = _linear_preprocessor() if model_name in LINEAR_MODELS else _tree_preprocessor()
    return Pipeline(
        [
            ("features", FunctionTransformer(build_features, validate=False)),
            ("preprocess", preprocess),
            ("model", _make_estimator(model_name, merged)),
        ]
    )


def plain_params(search_params: dict) -> dict:
    """RandomizedSearchCV names ('model__regressor__alpha') -> plain names ('alpha')."""
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
# Metrics and cross-validation
# --------------------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    """RMSE and MAE in bikes per hour, R2, and WAPE (total absolute error / total demand)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    total = float(np.sum(np.abs(y_true)))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - np.sum(error**2) / ss_tot) if ss_tot > 0 else float("nan"),
        "wape": float(np.sum(np.abs(error)) / total) if total > 0 else float("nan"),
    }


def predict_counts(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predictions clipped at 0: a rental count cannot be negative."""
    return np.clip(np.asarray(pipeline.predict(X), dtype=float), 0.0, None)


def clipped_rmse(y_true, y_pred) -> float:
    """RMSE of predictions clipped at 0, so that searches score exactly what the service returns."""
    error = np.asarray(y_true, dtype=float) - np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return float(np.sqrt(np.mean(error**2)))


# scikit-learn scorer for GridSearchCV / RandomizedSearchCV / permutation_importance (higher is better)
RMSE_SCORER = make_scorer(clipped_rmse, greater_is_better=False)


def summarise_folds(folds: list) -> dict:
    """Mean and standard deviation of every metric over the cross-validation folds."""
    summary = {}
    for metric in ("rmse", "mae", "r2", "wape"):
        values = np.array([f[metric] for f in folds], dtype=float)
        summary[f"{metric}_mean"] = float(np.nanmean(values))
        summary[f"{metric}_std"] = float(np.nanstd(values))
    return summary


def cross_validate_model(pipeline, X: pd.DataFrame, y: pd.Series, groups, n_splits: int = N_SPLITS, splitter=None) -> dict:
    """Cross-validation (blocked by default); returns per-fold metrics, mean/std and out-of-fold predictions.

    ``splitter`` can be replaced (for example by a shuffled KFold) to demonstrate leakage.
    """
    splitter = splitter if splitter is not None else GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    folds = []
    for train_idx, valid_idx in splitter.split(X, y, groups):
        model = clone(pipeline)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = predict_counts(model, X.iloc[valid_idx])
        oof[valid_idx] = pred
        folds.append(regression_metrics(y.iloc[valid_idx], pred))
    return {"folds": folds, "summary": summarise_folds(folds), "oof_pred": oof}


def profile_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Naive reference model: average demand of the same hour and day type (working day / day off)."""
    key_train = build_features(train_df[INPUT_COLUMNS])["hour_daytype"].to_numpy()
    key_test = build_features(test_df[INPUT_COLUMNS])["hour_daytype"]
    means = train_df[TARGET].groupby(key_train).mean()
    return key_test.map(means).fillna(float(train_df[TARGET].mean())).to_numpy(dtype=float)


def training_ranges(X: pd.DataFrame) -> dict:
    """Min and max of every weather input as seen in training (humidity zeros excluded)."""
    features = build_features(X)
    ranges = {}
    for column in WEATHER_COLUMNS:
        values = features[column].dropna()
        ranges[column] = [round(float(values.min()), 2), round(float(values.max()), 2)]
    return ranges


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
    """Validate one request record. Returns ``(clean_record, errors)``; ``clean_record`` is None on error."""
    if not isinstance(record, dict):
        return None, ["a record must be a JSON object"]
    errors = []
    unknown = sorted(set(record) - set(INPUT_SPEC))
    if unknown:
        errors.append("unknown field(s): " + ", ".join(unknown))
    missing = [name for name in INPUT_SPEC if name not in record]
    if missing:
        errors.append("missing field(s): " + ", ".join(missing))

    clean = {}
    for name, spec in INPUT_SPEC.items():
        if name not in record:
            continue
        value = record[name]
        kind = spec["kind"]
        if kind == "date":
            if not isinstance(value, str):
                errors.append("date must be a string in the format YYYY-MM-DD")
                continue
            try:
                parsed = datetime.strptime(value, "%Y-%m-%d").date()
            except ValueError:
                errors.append(f"date must be a valid calendar date in the format YYYY-MM-DD, got {value!r}")
                continue
            if not 1900 <= parsed.year <= 2100:
                errors.append("date must be between the years 1900 and 2100")
                continue
            clean[name] = parsed
        elif kind == "bool":
            if not isinstance(value, bool):
                errors.append(f"{name} must be true or false")
                continue
            clean[name] = value
        elif kind == "int":
            if not _is_real(value) or float(value) != int(value):
                errors.append(f"{name} must be an integer")
                continue
            number = int(value)
            if not spec["min"] <= number <= spec["max"]:
                errors.append(f"{name} must be between {spec['min']} and {spec['max']}, got {number}")
                continue
            clean[name] = number
        else:  # float
            if not _is_real(value):
                errors.append(f"{name} must be a number")
                continue
            number = float(value)
            if not spec["min"] <= number <= spec["max"]:
                unit = f" {spec['unit']}" if spec.get("unit") else ""
                errors.append(f"{name} must be between {spec['min']} and {spec['max']}{unit}, got {number}")
                continue
            clean[name] = number
    if errors:
        return None, errors
    return clean, []


def records_to_frame(records: list) -> pd.DataFrame:
    """List of validated records -> DataFrame with the columns the Pipeline expects."""
    frame = pd.DataFrame(records, columns=INPUT_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["hour"] = frame["hour"].astype(int)
    frame["holiday"] = frame["holiday"].astype(bool)
    for column in WEATHER_COLUMNS:
        frame[column] = frame[column].astype(float)
    return frame


def input_warnings(record: dict, ranges: dict | None) -> list:
    """Non-blocking remarks about a validated record."""
    warnings = []
    for column, (low, high) in (ranges or {}).items():
        value = record[column]
        if value < low or value > high:
            warnings.append(
                f"{column}={value} is outside the range seen in training ({low} to {high}); "
                "the prediction is an extrapolation"
            )
    if record["dew_point_c"] > record["temperature_c"] + 0.5:
        warnings.append("dew_point_c is higher than temperature_c, which is physically implausible")
    return warnings
