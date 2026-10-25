"""Flask prediction service for the Seoul bike-demand model.

    python predict.py                              # development server, http://localhost:9696
    gunicorn --bind 0.0.0.0:9696 predict:app       # production server (Linux, used in the Docker image)

Environment variables: MODEL_PATH (default: model.bin next to this file), PORT (default 9696),
HOST (default 127.0.0.1, development server only).

The service loads the whole scikit-learn Pipeline written by train.py, feature engineering
included, so a request needs only the raw inputs listed under GET /.
"""
from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request

import common

HERE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", HERE / "model.bin"))

# Loaded once at start-up: a missing or broken model file stops the service immediately
# instead of failing on the first request.
bundle = common.load_bundle(MODEL_PATH)
pipeline = bundle["pipeline"]
metadata = bundle["metadata"]

app = Flask(__name__)
app.json.sort_keys = False
app.config["MAX_CONTENT_LENGTH"] = 1_000_000  # bytes

UNIT = metadata.get("unit", "bikes per hour")
MODEL_NAME = metadata.get("model_name")
MAX_REPORTED_PROBLEMS = 20


def error_response(message: str, details=None, status: int = 400):
    body = {"error": message}
    if details:
        body["details"] = details
    return jsonify(body), status


def predict_values(records: list) -> list:
    frame = common.records_to_frame(records)
    return [round(float(value), 1) for value in common.predict_counts(pipeline, frame)]


def model_summary() -> dict:
    keys = ("model_name", "unit", "trained_at_utc", "fit_rows", "fit_date_range", "training_ranges", "holdout_test", "params_status")
    return {key: metadata[key] for key in keys if key in metadata}


@app.get("/")
def index():
    fields = {
        name: {"type": spec["kind"], **{k: v for k, v in spec.items() if k != "kind"}}
        for name, spec in common.INPUT_SPEC.items()
    }
    return jsonify(
        {
            "service": "seoul-bike-demand",
            "description": "Forecasts the number of rental bikes per hour in Seoul from calendar and weather inputs.",
            "model": model_summary(),
            "endpoints": {
                "GET /health": "liveness / readiness probe",
                "POST /predict": "one record -> predicted_rentals",
                "POST /predict_batch": f'{{"records": [...]}} with 1 to {common.MAX_BATCH_SIZE} records',
            },
            "input_fields": fields,
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.post("/predict")
def predict():
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return error_response("request body must be a valid JSON object")
    record, errors = common.validate_record(payload)
    if errors:
        return error_response("invalid input", errors)
    (value,) = predict_values([record])
    return jsonify(
        {
            "predicted_rentals": value,
            "unit": UNIT,
            "model": MODEL_NAME,
            "warnings": common.input_warnings(record, metadata.get("training_ranges")),
        }
    )


@app.post("/predict_batch")
def predict_batch():
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        return error_response('body must be a JSON object of the form {"records": [ ... ]}')
    extra = sorted(set(payload) - {"records"})
    if extra:
        return error_response("unknown field(s): " + ", ".join(extra))
    records = payload["records"]
    if not 1 <= len(records) <= common.MAX_BATCH_SIZE:
        return error_response(f"'records' must contain between 1 and {common.MAX_BATCH_SIZE} records")

    cleaned, problems = [], []
    for index, item in enumerate(records):
        record, errors = common.validate_record(item)
        if errors:
            problems.extend(f"record {index}: {message}" for message in errors)
        else:
            cleaned.append(record)
    if problems:
        if len(problems) > MAX_REPORTED_PROBLEMS:
            hidden = len(problems) - MAX_REPORTED_PROBLEMS
            problems = problems[:MAX_REPORTED_PROBLEMS] + [f"... and {hidden} more"]
        return error_response("invalid input", problems)

    values = predict_values(cleaned)
    ranges = metadata.get("training_ranges")
    return jsonify(
        {
            "count": len(values),
            "unit": UNIT,
            "model": MODEL_NAME,
            "predictions": [
                {"index": i, "predicted_rentals": value, "warnings": common.input_warnings(record, ranges)}
                for i, (record, value) in enumerate(zip(cleaned, values))
            ],
        }
    )


@app.errorhandler(404)
def not_found(_error):
    return error_response("not found", status=404)


@app.errorhandler(405)
def method_not_allowed(_error):
    return error_response("method not allowed for this endpoint", status=405)


@app.errorhandler(413)
def payload_too_large(_error):
    return error_response("request body too large", status=413)


@app.errorhandler(500)
def internal_error(_error):
    return error_response("internal server error", status=500)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "9696")), debug=False)
