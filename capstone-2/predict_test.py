#!/usr/bin/env python
"""Smoke-test a running prediction service (standard library only).

    python predict_test.py                         # http://localhost:9696
    python predict_test.py http://localhost:8080   # e.g. a kubectl port-forward

Sends valid requests (expecting HTTP 200) and invalid ones (expecting HTTP 400) and prints one
line per check. Exit code 0 means every check behaved as expected.
"""
from __future__ import annotations

import copy
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMEOUT = 30
CLASS_KEYS = ("no_readmission", "readmission_after_30_days", "readmission_within_30_days")


def call(base: str, method: str, path: str, payload=None, raw: bytes | None = None):
    """Return (status, parsed_json_or_None)."""
    data = raw if raw is not None else (None if payload is None else json.dumps(payload).encode("utf-8"))
    request = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read()
    try:
        return status, json.loads(body.decode("utf-8"))
    except ValueError:
        return status, None


class Checker:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        print(f"{'OK  ' if ok else 'FAIL'} {label}" + (f"  -> {detail}" if detail else ""))
        if not ok:
            self.failures += 1


def valid_prediction(body) -> bool:
    """Probabilities for the three classes that sum to 1, and a decision that is one of the classes."""
    if not isinstance(body, dict) or body.get("predicted_class") not in CLASS_KEYS:
        return False
    probabilities = body.get("probabilities", {})
    values = [probabilities.get(key) for key in CLASS_KEYS]
    return all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in values) and abs(sum(values) - 1.0) < 0.01


def summary(body) -> str:
    p = body["probabilities"]
    return (
        f"{body['predicted_label']} | P(none)={p['no_readmission']:.3f} P(>30)={p['readmission_after_30_days']:.3f} "
        f"P(<30)={p['readmission_within_30_days']:.3f}"
    )


def main(argv) -> int:
    base = (argv[1] if len(argv) > 1 else "http://localhost:9696").rstrip("/")
    sample = json.loads((HERE / "sample_request.json").read_text(encoding="utf-8"))
    checks = Checker()

    try:
        status, body = call(base, "GET", "/health")
    except (urllib.error.URLError, OSError) as error:
        print(f"Could not reach the service at {base} ({error}). Is it running?")
        return 1
    checks.check("GET /health", status == 200 and body and body.get("status") == "ok", str(body))

    status, body = call(base, "GET", "/")
    checks.check("GET /", status == 200 and body and "input_fields" in body,
                 f"model {body['model'].get('model_name')}" if status == 200 and body else str(status))

    status, body = call(base, "POST", "/predict", sample)
    ok = status == 200 and valid_prediction(body)
    checks.check("POST /predict (sample_request.json)", ok, summary(body) if ok else str(body))

    low = dict(sample, number_inpatient=0, number_emergency=0, number_outpatient=0, time_in_hospital=2,
               age="[40-50)", A1Cresult="None", diag_1="486", diag_2=None, diag_3=None, number_diagnoses=3,
               medications={"insulin": "No"}, change="No", num_medications=9, discharge_disposition_id=1)
    high = dict(sample, number_inpatient=6, number_emergency=3, number_outpatient=2, time_in_hospital=9,
                age="[80-90)", A1Cresult=">8", number_diagnoses=9, num_medications=28, discharge_disposition_id=3)
    risks = {}
    for label, record in (("low-utilisation patient", low), ("high-utilisation patient", high)):
        status, body = call(base, "POST", "/predict", record)
        ok = status == 200 and valid_prediction(body)
        risks[label] = body["early_readmission_risk"] if ok else None
        checks.check(f"POST /predict {label}", ok, summary(body) if ok else str(body))
    if all(v is not None for v in risks.values()):
        relation = ">" if risks["high-utilisation patient"] > risks["low-utilisation patient"] else "<="
        print(f"NOTE sanity: early-readmission risk of the high-utilisation patient {relation} the low-utilisation one "
              f"({risks['high-utilisation patient']:.3f} vs {risks['low-utilisation patient']:.3f}); more prior inpatient visits should raise the risk")

    minimal = {k: v for k, v in sample.items() if k not in ("payer_code", "medical_specialty", "diag_2", "diag_3", "medications",
                                                            "max_glu_serum", "A1Cresult")}
    status, body = call(base, "POST", "/predict", minimal)
    checks.check("POST /predict with optional fields omitted", status == 200 and valid_prediction(body),
                 summary(body) if status == 200 and valid_prediction(body) else str(body))

    odd = dict(sample, medical_specialty="NeverSeenBefore", payer_code="ZZ", time_in_hospital=30)
    status, body = call(base, "POST", "/predict", odd)
    checks.check("POST /predict unusual values give warnings, not errors",
                 status == 200 and valid_prediction(body) and len(body.get("warnings", [])) >= 1,
                 f"{len(body.get('warnings', []))} warning(s)" if status == 200 and body else str(body))

    records = [dict(sample, number_inpatient=n) for n in range(5)]
    status, body = call(base, "POST", "/predict_batch", {"records": records})
    ok = status == 200 and body is not None and body.get("count") == 5 and all(valid_prediction(p) for p in body["predictions"])
    checks.check("POST /predict_batch (5 records)", ok,
                 "early risk by prior inpatient visits: " + ", ".join(f"{p['early_readmission_risk']:.3f}" for p in body["predictions"]) if ok else str(body))

    bad_cases = {
        "age label not in the dataset": dict(sample, age="70-80"),
        "expired patient (discharge 11)": dict(sample, discharge_disposition_id=11),
        "hospice discharge (13)": dict(sample, discharge_disposition_id=13),
        "missing field": {k: v for k, v in sample.items() if k != "diag_1"},
        "unknown field": dict(sample, race="Other"),
        "invalid ICD-9 code": dict(sample, diag_1="25O.83"),
        "unknown drug": dict(sample, medications={"aspirin": "Steady"}),
        "invalid drug status": dict(sample, medications={"insulin": "Increased"}),
        "negative visit count": dict(sample, number_inpatient=-1),
        "number sent as text": dict(sample, time_in_hospital="five"),
        "boolean sent as number": dict(sample, time_in_hospital=True),
        "lab result not allowed": dict(sample, A1Cresult="high"),
    }
    for label, record in bad_cases.items():
        status, body = call(base, "POST", "/predict", copy.deepcopy(record))
        checks.check(f"invalid input rejected: {label}", status == 400 and body is not None and "error" in body,
                     f"{status} {body.get('details') if body else ''}")

    status, body = call(base, "POST", "/predict", raw=b"this is not json")
    checks.check("invalid input rejected: not JSON", status == 400, str(status))
    status, body = call(base, "POST", "/predict_batch", {"records": []})
    checks.check("empty batch rejected", status == 400, str(status))
    status, body = call(base, "GET", "/nothing-here")
    checks.check("unknown route returns 404", status == 404, str(status))

    print()
    if checks.failures:
        print(f"{checks.failures} check(s) failed.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
