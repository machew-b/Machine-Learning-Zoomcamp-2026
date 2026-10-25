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
    ok = status == 200 and body is not None and body.get("predicted_rentals", -1) >= 0
    checks.check("POST /predict (sample_request.json)", ok, f"{body.get('predicted_rentals')} {body.get('unit')}" if ok else str(body))

    forecasts = {}
    for hour in (3, 8, 13, 18):
        record = dict(sample, hour=hour, solar_radiation_mj_m2=0.0 if hour in (3, 18) else 1.2)
        status, body = call(base, "POST", "/predict", record)
        ok = status == 200 and body is not None
        forecasts[hour] = body["predicted_rentals"] if ok else None
        checks.check(f"POST /predict hour={hour:02d} (weekday)", ok, str(forecasts[hour]))
    if all(v is not None for v in forecasts.values()):
        print(f"NOTE sanity: 18:00 forecast {'>' if forecasts[18] > forecasts[3] else '<='} 03:00 forecast "
              f"(a commuting evening should be busier than the middle of the night)")

    rainy = dict(sample, rainfall_mm=8.0, humidity_pct=92, solar_radiation_mj_m2=0.0, visibility_10m=600)
    status, body = call(base, "POST", "/predict", rainy)
    checks.check("POST /predict rainy evening", status == 200 and body is not None,
                 str(body.get("predicted_rentals")) if status == 200 and body else str(body))

    sunday = dict(sample, date="2018-07-22", hour=15, solar_radiation_mj_m2=2.1)
    status, body = call(base, "POST", "/predict", sunday)
    checks.check("POST /predict Sunday afternoon", status == 200 and body is not None,
                 str(body.get("predicted_rentals")) if status == 200 and body else str(body))

    records = [dict(sample, hour=h, solar_radiation_mj_m2=max(0.0, 1.4 - abs(h - 13) * 0.2)) for h in range(24)]
    status, body = call(base, "POST", "/predict_batch", {"records": records})
    ok = status == 200 and body is not None and body.get("count") == 24
    checks.check("POST /predict_batch (24 hours)", ok,
                 "peak hour " + str(max(body["predictions"], key=lambda p: p["predicted_rentals"])["index"]) if ok else str(body))

    bad_cases = {
        "hour out of range (24)": dict(sample, hour=24),
        "missing field": {k: v for k, v in sample.items() if k != "temperature_c"},
        "unknown field": dict(sample, season="Summer"),
        "invalid date": dict(sample, date="2018-02-30"),
        "humidity above 100": dict(sample, humidity_pct=120),
        "number sent as text": dict(sample, temperature_c="hot"),
        "boolean expected": dict(sample, holiday="no"),
        "negative rainfall": dict(sample, rainfall_mm=-1),
    }
    for label, record in bad_cases.items():
        status, body = call(base, "POST", "/predict", copy.deepcopy(record))
        checks.check(f"invalid input rejected: {label}", status == 400 and body is not None and "error" in body,
                     f"{status} {body.get('details') if body else ''}")

    status, body = call(base, "POST", "/predict", raw=b"this is not json")
    checks.check("invalid input rejected: not JSON", status == 400, str(status))
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
