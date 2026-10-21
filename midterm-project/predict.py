"""Flask web service that serves the purchase-intent model.

Local development:   python predict.py
Production (Docker): gunicorn --bind 0.0.0.0:9696 predict:app

Endpoints:
    GET  /          service + model information
    GET  /health    liveness / readiness probe
    POST /predict   JSON body with one browsing session -> purchase probability
"""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

import common  # noqa: F401  (also needed so the pickled pipeline can be loaded)

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(common.MODEL_PATH)))

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Run `python train.py` first.")
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

pipeline = bundle["pipeline"]
THRESHOLD = float(bundle["threshold"])
MODEL_NAME = bundle["model_name"]

app = Flask("shopper-intent")


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def _number(payload: dict, name: str, minimum: float = 0.0, maximum: float | None = None,
            integer: bool = False):
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a number")
    if integer and value != int(value):
        raise ValueError(f"{name} must be a whole number")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum:g}")
    return int(value) if integer else float(value)


def _boolean(payload: dict, name: str) -> bool:
    value = payload[name]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{name} must be true or false")


def parse_session(payload) -> tuple[pd.DataFrame | None, list[str]]:
    """Validate one session and turn it into a one-row DataFrame."""
    if not isinstance(payload, dict):
        return None, ["Request body must be a JSON object with the session features."]

    missing = [c for c in common.RAW_FEATURES if c not in payload]
    if missing:
        return None, [f"Missing fields: {', '.join(missing)}"]

    row, errors = {}, []

    def collect(name: str, parser, *args, **kwargs) -> None:
        try:
            row[name] = parser(payload, name, *args, **kwargs)
        except ValueError as exc:
            errors.append(str(exc))

    for col in common.COUNT_COLUMNS:
        collect(col, _number, integer=True)
    for col in common.DURATION_COLUMNS + common.VALUE_COLUMNS:
        collect(col, _number)
    for col in common.RATE_COLUMNS:
        collect(col, _number, maximum=1.0)
    for col in common.CODED_COLUMNS:
        collect(col, _number, minimum=1, integer=True)
    collect("Weekend", _boolean)

    month = common.normalize_month(payload["Month"])
    if month in common.MONTHS:
        row["Month"] = month
    else:
        errors.append(f"Month must be a month name such as 'Feb' or 'June', got {payload['Month']!r}")

    if payload["VisitorType"] in common.VISITOR_TYPES:
        row["VisitorType"] = payload["VisitorType"]
    else:
        errors.append(f"VisitorType must be one of {common.VISITOR_TYPES}")

    if errors:
        return None, errors
    return pd.DataFrame([row], columns=common.RAW_FEATURES), []


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return jsonify(
        service="shopper-intent",
        model=MODEL_NAME,
        threshold=round(THRESHOLD, 4),
        endpoints={"GET /health": "health check", "POST /predict": "score one browsing session"},
        required_fields=common.RAW_FEATURES,
    )


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="Body must be valid JSON (Content-Type: application/json)."), 400

    session, errors = parse_session(payload)
    if errors:
        return jsonify(error="Invalid input", details=errors), 400

    probability = float(pipeline.predict_proba(session)[0, 1])
    return jsonify(
        purchase_probability=round(probability, 4),
        will_purchase=bool(probability >= THRESHOLD),
        threshold=round(THRESHOLD, 4),
        model=MODEL_NAME,
    )


if __name__ == "__main__":
    # Development server only; Docker / Kubernetes use gunicorn (see Dockerfile).
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "9696")), debug=False)
